from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import time

llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    api_key = "vllm-token",
    model="gemma-3-27b-it.gguf"
)

promptA = ChatPromptTemplate.from_messages([
    ("system","你是熱情的網紅，以富有創意的方式回應。 (20字內回應)"),
    ("human","{article_content}")
    ])

promptB = ChatPromptTemplate.from_messages([
    ("system","你是頑固的企業家，以嚴肅的方式回應 (20字內回應)"),
    ("human","{article_content}")
    ])

parser = StrOutputParser()

chainA = promptA | llm.bind(max_tokens = 25,temperature = 0) | parser

chainB = promptB | llm.bind(max_tokens = 25,temperature = 0) | parser

runnable_chain = RunnableParallel(
    styleA =chainA,
    styleB = chainB
)

user_input = input("User: ")

print("---開始回應 (串流模式)---")
for chunk in runnable_chain.stream({"article_content": user_input}):
    print(chunk,end="\n",flush=True)

start_time = time.perf_counter()

print("--- 開始回應 (批次處理)---")
result = runnable_chain.batch([{"article_content": user_input}])

end_time = time.perf_counter()

response_time = end_time - start_time
print(f"花費時間 : {response_time:.2f} 秒\n")

print(f"styleA: {result[0]['styleA']}")
print(f"styleB: {result[0]['styleB']}")
