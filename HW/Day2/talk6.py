from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
import json 

llm = ChatOpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token",
    model="Qwen/Qwen3-VL-8B-Instruct"
)

system_prompt = "你是一個資料提取助手。{format_instructions} 需要的欄位: name, phone, product, quantity, address"

prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{text}")
    ])

parser = JsonOutputParser()

chain = prompt | llm | parser

try:
    tech_article = "你好，我是陳大明，電話是 0912-345-678，\
    我想要訂購 3 台筆記型電腦，下週五送到台中市北區"

    print("--- Wait... ---")
    result = chain.invoke({"article_content": tech_article})
    print(result)
except Exception as e:
    print(f"Error {e}")
