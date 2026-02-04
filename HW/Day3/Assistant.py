from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END
import time
import requests
from pathlib import Path

BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
WAV_PATH = "./Podcast_EP14_30s.wav" # 請自行更改為測試音檔路徑
auth = ("nutc2504", "nutc2504")

out_dir = Path("./out")
out_dir.mkdir(exist_ok=True)

# 1) 建立任務
with open(WAV_PATH, "rb") as f:
    r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
r.raise_for_status()
task_id = r.json()["id"]
print("task_id:", task_id)
print("等待轉文字...")
txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT" 
srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

def wait_download(url: str, max_tries=600):  # 等下載完成
    for _ in range(max_tries):
        try:
            resp = requests.get(url, timeout=(5, 60), auth=auth)
            if resp.status_code == 200:
                return resp.text
            # 還沒好通常 404
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(2)
    return None

# 2) 等 TXT(純文字)
txt_text = wait_download(txt_url, max_tries=600)
if txt_text is None:
    raise TimeoutError("轉錄逾時or錯誤")
    
# 3) 等 SRT(有時間軸+文字)
srt_text = wait_download(srt_url, max_tries=600)

# 4) 存檔（完整）
txt_path = out_dir / f"{task_id}.txt"
txt_path.write_text(txt_text, encoding="utf-8")
print("轉錄成功:", txt_path)

if srt_text is not None:
    srt_path = out_dir / f"{task_id}.srt"
    srt_path.write_text(srt_text, encoding="utf-8")
    print("轉錄成功:", srt_path)


llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    api_key = "vllm-token",
    model="google/gemma-3-27b-it",
    temperature= 0
)
#元件1 狀態

class AgentState(TypedDict):
    asr_text: str
    raw_text: str
    minutes_result: str
    summary_result: str
    final_output: str

#元件2 節點
#Node asr節點
def asr_node(state: AgentState):
    print("---正在載入asr---")
    return {"asr_text": srt_text,"raw_text":txt_text}
#Node minutes_taker節點
def minutes_taker_node(state: AgentState):
    """
    SRT總結工具
    """
    print("---Node: minutes_taker (整理)")

    prompt = [
        SystemMessage(content="你是專業的會議紀錄員。根據提供的 SRT 時間軸字幕，整理出一份詳細的「逐字稿紀錄」。格式須包含時間點和內容"),
        HumanMessage(content=f"字幕內容：\n{state['asr_text']}")        
    ]
    response = llm.invoke(prompt)
    return {"minutes_result": [response.content]}

#Node summarizer節點
def summarizer_node(state: AgentState):
    """
    大剛彙整工具
    """
    print("---Node: summarizer(重點)")

    prompt = [
        SystemMessage(content="你是高效率的分析師。閱讀會議內容，列舉出3~5個關鍵重點摘要"),
        HumanMessage(content=f"會議內容：\n{state['raw_text']}")        
    ]
    response = llm.invoke(prompt)
    return {"summary_result": [response.content]}
    
#Node 輸出
def writer_node(state: AgentState):
    """
    彙整兩節點最終結果
    """
    print("---Node: writer最終結果)")

    minutes = state.get('minutes_result')[0]

    summary = state.get('summary_result')[0]

    final_doc = (
        "##重點\n"
        f"{summary}\n\n"
        "-------------\n\n"
        "##逐字紀錄\n"
        f"{minutes}\n"
    )

    return {"final_output": final_doc}

#組裝
workflow = StateGraph(AgentState)

workflow.add_node("asr",asr_node)

workflow.add_node("minutes_taker",minutes_taker_node)

workflow.add_node("summarizer",summarizer_node)

workflow.add_node("writer",writer_node)

workflow.set_entry_point("asr")

workflow.add_edge("asr","minutes_taker")

workflow.add_edge("asr","summarizer")

workflow.add_edge("minutes_taker","writer")

workflow.add_edge("summarizer","writer")

workflow.add_edge("writer",END)

app = workflow.compile()

print(app.get_graph().draw_ascii())

#測試執行

if(__name__ == "__main__"):
    print("\n執行工作流...\n")

    final_state = app.invoke({})

    print("---執行結束---")

    print(final_state["final_output"])