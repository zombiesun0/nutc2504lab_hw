from openai import OpenAI

client = OpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token"
)

history = [{"role": "system","content":"you are an AI"}]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit","q"]:
        print("Quit")
        break
    
    history.append({"role": "user","content": user_input})
    
    try:
        print("Now AI is thinking...",end="\r")
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=history,
            temperature=0.7,
            max_tokens=100
        )
        fully_reply = response.choices[0].message.content
        print(f"AI : {fully_reply}\n")
        history.append({"role": "assistant","content": fully_reply})
    except Exception as e:
        print(f"error {e}")
