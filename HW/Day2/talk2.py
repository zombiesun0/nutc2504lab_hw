from openai import OpenAI

client = OpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    api_key = "vllm-token"
)

prompt = "introduce yourself"
temps = [0.1,1.5]

for t in temps:
    print(f"\n try temperature = {t} ...")
    try:
        response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "user","content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
        )
        print(f"answer {response.choices[0].message.content}")
    except Exception as e:
        print(f"error {e}")
    
