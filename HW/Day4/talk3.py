import json
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,ToolMessage

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
def extract_order_data(name: str,phone: str,product: str,quantity: int, address: str):
    """
    資料提取專用工具
    專門用於從非結構化文本中提取訂單相關資訊 （姓名、電話、商品、數量、地址）
    """

    return{
        "name" : name,
        "phone" : phone,
        "product" : product,
        "quantity" : quantity,
        "address" : address
    }

VIP_LIST = ["Link","Mario"]

llm_with_tools = llm.bind_tools([extract_order_data])

tool_node = ToolNode([extract_order_data])
#元件1 狀態

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
                 

#元件2 節點
#Node 思考
def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
#Node 審核 
def human_review_node(state: AgentState):
    """
    如偵測到VIP等人工輸入
    """
    print("\n" + "="*30)

    print("VIP觸發")

    print("="*30)

    last_msg = state['messages'][-1]

    print(f"等待審核資料: {last_msg.content}")

    review = input("請輸入 yes/no>>>")

    if(review.lower() == "yes"):
        return{
            "messages": [
                AIMessage(content="已收到訂單資料，因偵測到 VIP 客戶，系統將轉交人工審核..."),
                HumanMessage(content="[系統公告] 管理員已人工審核通過此 VIP 訂單，請繼續完成後續動作。")
            ]
        }
    else:
        return{
            "messages": [
                AIMessage(content="已收到訂單資料，等待人工審核結果..."),
                HumanMessage(content="[系統公告] 管理員拒絕了此訂單，請取消交易並告知用戶。")
            ]
        }
#元件3 邊緣決策
def entry_router(state: AgentState):
    last_message = state["messages"][-1]
    if(last_message.tool_calls):
        return "tools"
    return END

def post_tool_router(state: AgentState) -> Literal["human_review","agent"]:
    messages = state["messages"]
    last_messages = messages[-1]

    if(isinstance(last_messages,ToolMessage)):
        try:
            data = json.loads(last_messages.content)
            user_name = data.get("name","")

            if(user_name in VIP_LIST):
                print(f"DEBUG: 發現VIP [{user_name}] -> 人工審查")
                return "human_review"
        except Exception as e:
            print(e)

    return "agent"
#組裝
workflow = StateGraph(AgentState)

workflow.add_node("agent",agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("human_review",human_review_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    entry_router,
    {"tools": "tools",END: END}
)

workflow.add_conditional_edges(
    "tools",
    post_tool_router,
    {
        "human_review": "human_review",
        "agent": "agent"
    }
)

workflow.add_edge("human_review","agent")

app = workflow.compile()

print(app.get_graph().draw_ascii())

#測試執行

if(__name__ == "__main__"):
    print(f"VIP 名單 {VIP_LIST}")

    while True:
        user_input = input("User: ")
        if(user_input.lower() in ["exit","q"]): 
            break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if(key == "agent"):
                    msg = value["messages"][-1]
                    if(not msg.tool_calls):
                        print(f"-> [Agent]: {msg.content}")
                elif key == "human_review":
                    print("審核完成")
