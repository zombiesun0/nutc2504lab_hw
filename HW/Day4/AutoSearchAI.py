import os
import json
import operator
from typing import TypedDict, List, Annotated

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from search_searxng import search_searxng
from vlm_read_website import vlm_read_website

# 初始化 LLM
llm = ChatOpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token",
    model="Qwen3-VL-8B-Instruct-BF16.gguf",
    temperature= 0
)

CACHE_FILE = "qa_cache.json"

# ================= 快取與工具函式 =================

def get_clean_key(text: str) -> str:
    return text.replace(" ", "").replace("?", "")

def load_cache():
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_cache(new_data: dict):
    current = load_cache()
    current.update(new_data)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=4)

# ================= 定義狀態 (State) =================

class AgentState(TypedDict):
    question: str
    answer: str
    knowledge_base: Annotated[List[str], operator.add] # 累積有價值的資訊
    current_query: str 
    loop_count: int
    source: str # CACHE / SEARCH / PLANNER

# ================= 定義節點 (Nodes) =================

def check_cache_node(state: AgentState):
    """快取檢查"""
    print(f"\n[1] 檢查快取: {state['question']}")
    cache = load_cache()
    key = get_clean_key(state['question'])
    
    if key in cache:
        print("   -> 命中快取 (Hit)")
        return {"answer": cache[key], "source": "CACHE"}
    else:
        print("   -> 快取未命中 (Miss)")
        # 初始化 knowledge_base 為空列表
        return {"source": "SEARCH", "knowledge_base": [], "loop_count": 0}

def planner_node(state: AgentState):
    """規劃器：決定繼續搜尋或回答"""
    print(f"[2] Planner 評估中 (Loop: {state.get('loop_count', 0)})...")
   
    if state.get('loop_count', 0) >= 3: # 安全機制：超過 3 次強制回答
        print("   -> 達最大迴圈數，強制回答。")
        return {"source": "force_answer"}

    if not state.get('knowledge_base'):  # 如果完全沒知識
        print("   -> 尚無資訊，需要搜尋。")
        return {"source": "need_search"}
   
    context_str = "\n".join(state['knowledge_base']) # 讓 LLM 判斷資訊足夠與否
    prompt = f"""
    問題: {state['question']}
    目前已知資訊:
    {context_str}
    
    請問上述資訊是否已經足夠完整回答問題？
    回答 "YES" 表示足夠，回答 "NO" 表示需要更多資訊。
    """
    judge = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    
    if "YES" in judge:
        print("   -> 資訊充足，準備回答。")
        return {"source": "ready_to_answer"}
    else:
        print("   -> 資訊不足，繼續搜尋。")
        return {"source": "need_search"}

def query_gen_node(state: AgentState):
    """生成關鍵字"""
    print("[3] 生成搜尋關鍵字...")
    prompt = f"基於問題 '{state['question']}' 與已知資訊，生成一個最重要的搜尋關鍵字。"
    query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print(f"   -> 關鍵字: {query}")
    
    return {"current_query": query, "loop_count": state.get("loop_count", 0) + 1}

def search_node(state: AgentState):
    """搜尋 + VLM 讀取"""
    query = state.get("current_query", state["question"])
    print(f"[4] 執行搜尋: {query}")
    
    # 1. 執行搜尋
    results = search_searxng(query=query, limit=2) 
    
    print("    -> VLM 讀取網頁並評估價值...")
    new_knowledge = []
    
    for res in results:
        url = res.get("url")
        title = res.get("title", "網頁")
        
        try:          
            content = vlm_read_website(url, title) #呼叫 VLM 讀取
            
            # 評估價值
            check_prompt = f"""
            問題: {state['question']}
            網頁內容: {content[:1000]}... (略)
            
            這段內容對回答問題有幫助嗎？有價值請回答 YES，否則回答 NO。
            """
            valuable = llm.invoke([HumanMessage(content=check_prompt)]).content.strip().upper()
                        
            if "YES" in valuable: # 4. 若有價值 -> 加入列表
                print(f"       [V] 發現有價值資訊: {title}")
                summary = f"來源 {title}: {content[:300]}..." 
                new_knowledge.append(summary)
            else:
                print(f"       [X] 資訊關聯度低: {title}")
        except Exception as e:
            print(f"       [!] 讀取失敗 {url}: {e}")
            
    # LangGraph 會自動 operator.add 將這裡回傳的 list 與原本的 knowledge_base 相加
    return {"knowledge_base": new_knowledge}

def final_node(state: AgentState):
    """生成最終回答並寫入快取"""
    print("[5] 生成最終回答...")
   
    if state.get("source") == "CACHE":  # 如果是從 Cache 來的 
        return {}

    context = "\n".join(state.get('knowledge_base', []))
    prompt = f"""
    請根據以下收集到的資訊回答問題：
    問題: {state['question']}
    資訊: {context}
    """
    final_ans = llm.invoke([HumanMessage(content=prompt)]).content
    
    # 寫入快取
    save_cache({get_clean_key(state['question']): final_ans})
    print("   -> 已更新快取。")
    
    return {"answer": final_ans}

# ================= 構建 =================

workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_node)
workflow.add_node("final_answer", final_node) 

workflow.set_entry_point("check_cache")

#Cache -> Planner 或 End
def route_cache(state):
    if state["source"] == "CACHE": return "end_flow" 
    return "planner"

workflow.add_conditional_edges(
    "check_cache",
    route_cache,
    {
        "end_flow": END,
        "planner": "planner"
    }
)

# Planner -> Final Answer 或 Query Gen
def route_planner(state):

    if state["source"] == "ready_to_answer" or state["source"] == "force_answer":
        return "final_answer" 
    return "query_gen"

workflow.add_conditional_edges(
    "planner", 
    route_planner,
    {
        "final_answer": "final_answer",
        "query_gen": "query_gen"
    }
)

# 建立循環
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")

# 結束
workflow.add_edge("final_answer", END) # [修正] 使用 final_answer

# 編譯
app = workflow.compile()

# ================= 執行 =================
if __name__ == "__main__":
    print(app.get_graph().draw_ascii()) 
    
    while True:
        user_q = input("\n請輸入問題 (q 離開): ")
        if user_q.lower() == "q": break

        inputs = {"question": user_q}
        try:
            result = app.invoke(inputs)
            print(f"\n💡 最終回答: {result['answer']}")
        except Exception as e:
            print(f"發生錯誤: {e}")