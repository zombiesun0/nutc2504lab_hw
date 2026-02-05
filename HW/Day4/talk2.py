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

#狀態
class State(TypedDict):
    original_text: str   #原始文字
    translated_text: str #翻譯結果
    critique: str        #評語
    attempts: int = 0       #重試次數

def translator_node(state: State):
    """翻譯節點"""
    print(f"\n---翻譯嘗試 (第 {state['attempts'] + 1} 次)---")

    prompt = f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋: '{state['original_text']}'"

    if(state["critique"]):
        prompt += f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋: '{state['original_text']}'\n\n上一輪的審查意見是: {state['critique']}。請根據意見修正翻譯。"
    
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "translated_text" : response.content,
        "attempts": state["attempts"] + 1
    }

def reflector_node(state: State):
    """負責審查節點"""
    print("---審查中---")
    print(f"翻譯: {state['translated_text']}")

    prompt = f"""
    你是一個嚴格的翻譯審查員。
    原文: {state['original_text']}
    翻譯: {state['translated_text']}

    請檢查翻譯是否準確且通順。
    - 如果翻譯很完美，請只回覆 "PASS"。
    - 如果需要修改，請給出簡短的具體建議。
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"critique": response.content}
#邊緣
def should_continue(state: State) -> Literal["translator","end"]:
    critique = state["critique"].strip().upper()

    if("PASS" in critique):
        print("---審查通過---")
        return "end"
    elif(state["attempts"] >= 3):
        print("---已達最大嘗試次數，停止---")
    else:
        print(f"---審查未通過: {state['critique']}---")
        print("---：退回重寫---")
        return "translator"
#組裝

workflow = StateGraph(State)

workflow.add_node("translator", translator_node)

workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("translator")

workflow.add_edge("translator","reflector")

workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "translator": "translator",
        "end": END
    }
)

app = workflow.compile()

print(app.get_graph().draw_ascii())

if(__name__ == "__main__"):
    while True:
        user_input = input("\nUser: ")
        if(user_input.lower() in ["exit","q"]): break
        inputs = {
            "original_text": user_input,
            "critique" : "",
            "attempts": 0
        }

        result = app.invoke(inputs)

        print("\n====== 結果 ======")
        print(f"原文: {result['original_text']}")
        print(f"翻譯: {result['translated_text']}")
        print(f"嘗試次數: {result['attempts']}")
