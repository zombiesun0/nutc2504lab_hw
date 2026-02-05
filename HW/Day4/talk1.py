import random
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage,ToolMessage

from langgraph.graph import StateGraph, END,add_messages
from langgraph.prebuilt import ToolNode
from typing import Literal

llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    api_key = "vllm-token",
    model="google/gemma-3-27b-it",
    temperature= 0
)

@tool
def get_weather(city: str):
    """
    查詢指定城市的天氣。輸入參數 city 必須是城市名稱。
    """
    if(random.random() < 0.5):
        return "資料庫連線失敗"

    if("台北" in city):
        return "台北下大雨 氣溫 15 度"
    elif("台中" in city):
        return "台中晴天 氣溫 22 度"
    elif("高雄" in city):
        return "高雄多雲 氣溫 17 度"
    else:
        return "資料庫找不到該城市資料"

tools = [get_weather]

llm_with_tools = llm.bind_tools(tools)

#元件1 狀態

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
                 

#元件2 節點
#Node A
def chatbot_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node_executor = ToolNode(tools)

def fallback_node(state: AgentState):
    last_message = state["messages"][-1]

    tool_call_id = last_message.tool_calls[0]["id"]

    error_message = ToolMessage(
        content = "已達到最大嘗試次數，停止嘗試，告訴使用者暫時無法使用服務",
        tool_call_id = tool_call_id
    )

    return {"message": [error_message]}
#元件4 定義邊
def router(state: AgentState) -> Literal["tools",'end']:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    
    retry_count = 0

    for msg in reversed(messages[:-1]):
        if isinstance(msg,ToolMessage):
            if "系統錯誤" in msg.content:
                retry_count += 1
            else:
                break
        elif isinstance(msg,HumanMessage):
            break
    
    print(f"DEBUG 重新嘗試次數 {retry_count}")

    if retry_count >= 3:
        return "fallback"

    return "tools"

#組裝
workflow = StateGraph(AgentState)

workflow.add_node("agent",chatbot_node)
workflow.add_node("tools", tool_node_executor)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools" : "tools",
        "fallback" : "fallback",
        "end" : END
    }
)

workflow.add_edge("tools","agent")

workflow.add_edge("fallback","agent")

app = workflow.compile()

print(app.get_graph().draw_ascii())

#測試執行

if(__name__ == "__main__"):
    while True:
        user_input = input("User: ")
        if(user_input.lower() in ["exit","q"]): 
            break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if(key == "agent"):
                    msg = value["messages"][-1]
                    if(msg.tool_calls):
                        print("Agent 呼叫工具中...")
                    else:
                        print(f"Agent: {msg.content}")
                elif key == "tools":
                    print("Tools 系統錯誤...")

                elif key == "fallback":
                    print("Fallback 觸發熔斷機制")