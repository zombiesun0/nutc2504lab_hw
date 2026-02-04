from openai import OpenAI

client = OpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token"
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit","q"]:
        print("Quit")
        break
    
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "system","content":"you are an AI"},
            {"role": "user","content": user_input}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"AI : {response.choices[0].message.content}")
