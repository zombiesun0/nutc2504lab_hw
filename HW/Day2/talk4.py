from openai import OpenAI

import json

client = OpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token"
)


user_input = "Hello, i'm Steve, My phone number is 0912-345-678,and I want buy 3 computer,send these to Street-A in next Friday"

system_prompt = """you are a Data Extraction Assistant. You need to Strictly enforced the return of data in JSON format. needed slots:
name,phone,product,quantity,address"""

response = client.chat.completions.create(
    model="google/gemma-3-27b-it",
    messages=[
        {"role": "system","content": system_prompt},
        {"role": "user","content": user_input}
    ],
    temperature=0.1,
)

json_content = response.choices[0].message.content

clean_json = json_content.replace("```json","").replace("```","").strip()

decision = json.loads(clean_json)

print(json.dumps(decision,ensure_ascii=False,indent=2))
