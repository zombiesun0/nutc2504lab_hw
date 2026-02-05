from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage,ToolMessage
from langgraph.graph import StateGraph, END,add_messages
from langgraph.prebuilt import ToolNode
from typing import Literal
import os
import json
import time

llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    api_key = "vllm-token",
    model="google/gemma-3-27b-it",
    temperature= 0.5
)

fast_llm = ChatOpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token",
    model="Qwen3-VL-8B-Instruct-BF16.gguf",
    temperature= 0
)

# 設定快取檔案名稱
CACHE_FILE = "qa_cache.json"

def get_clean_key(text: str) -> str:
    """統一將問題標準化"""
    return text.replace(" ", "").replace("?", "")

def load_cache():
    """從 JSON 讀取快取資料"""
    if not os.path.exists(CACHE_FILE):
        default_data = {
            get_clean_key("LangGraph是什麼"): "LangGraph 是一個用於構建有狀態、多參與者應用程式的框架。",
            get_clean_key("你的名字"): "我是這個課程的 AI 助教。"
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)
        return default_data

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}
    
def save_cache(new_data: dict):
    """將資料寫入 JSON"""
    current_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except:
            pass
    
    current_data.update(new_data)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

# 1. 定義狀態
class State(TypedDict):
    question: str
    answer: str
    source: str # CACHE / FAST / LLM

def check_cache_node(state: State):
    """檢查快取"""
    print(f"\n[系統] 收到問題: {state['question']}")
    cache_data = load_cache()
    clean_query = get_clean_key(state['question'])

    if clean_query in cache_data:
        print("--- 命中快取 (Cache Hit) ---")
        return {
            "answer": cache_data[clean_query],
            "source": "CACHE"
        }
    else:
        print("--- 快取未命中 (Cache Miss) ---")
        return {"source": "MISS"}

def fast_reply_node(state: State):
    print("--- 進入快速通道 (Fast Track API) ---")
    
    response = fast_llm.invoke([HumanMessage(content=state['question'])])
    
    return {
        "answer": response.content,
        "source": "FAST_TRACK_API"
    }

def expert_node(state: State):
    """慢速通道：呼叫 LLM 並使用流式傳輸"""
    print("--- 進入專家模式 (LLM Expert) ---")
    
    prompt = f"請以專業的角度回答以下問題：{state['question']}"
    chunks = llm.stream([HumanMessage(content=prompt)])
    
    full_answer = ""
    print("AI 正在思考並打字：", end="", flush=True)
    
    for chunk in chunks:
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_answer += content
    print("\n")
    
    # 存入快取
    clean_key = get_clean_key(state['question'])
    save_cache({clean_key: full_answer})
    print(f"--- [系統] 已將完整回答寫入 {CACHE_FILE} ---")
    
    return {
        "answer": full_answer,
        "source": "LLM_EXPERT"
    }

def master_router(state: State):
    """主路由控制器"""
    if state.get("answer"):
        return "end"
    
    question = state['question']
    # 只要包含這些打招呼的詞，就走 Fast Track
    if any(word in question for word in ["你好", "嗨", "早安", "哈囉"]):
        return "fast"
    else:
        return "expert"
    
workflow = StateGraph(State)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("fast_bot", fast_reply_node)
workflow.add_node("expert_bot", expert_node)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    master_router,
    {
        "end": END,
        "fast": "fast_bot",
        "expert": "expert_bot"
    }
)

workflow.add_edge("fast_bot", END)
workflow.add_edge("expert_bot", END)

app = workflow.compile()
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    print(f"快取檔案將儲存於: {os.path.abspath(CACHE_FILE)}")

    while True:
        user_input = input("\n請輸入問題 (輸入 q 離開): ")
        if user_input.lower() == 'q':
            break
            
        inputs = {"question": user_input}
        
        start_time = time.time()
        try:
            result = app.invoke(inputs)
            end_time = time.time()
            
            print("-" * 30)
            print(f"來源: {result['source']}")
            print(f"耗時: {end_time - start_time:.4f} 秒")
            
            if result['source'] != "LLM_EXPERT":
                print(f"回答: {result['answer']}")
            else:
                print("(回答輸出完畢)")
                
        except Exception as e:
            print(f"發生錯誤: {e}")