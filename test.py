from transformers import AutoTokenizer, pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"  # 用指令微调版

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    device_map="auto",         # 有多卡/单卡会自动放
    torch_dtype="auto"
)

def chat_once(messages):
    """
    messages: 形如 [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 自动加上 assistant 的开头标记
    )
    out = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )[0]["generated_text"]

    # 简单做法：把原始 prompt 切掉，只保留新生成的部分
    reply = out[len(prompt):]
    return reply.strip()

history = [
    {"role": "system", "content": "You are a helpful AI assistant that replies in Chinese."}
]

while True:
    user_input = input("你：")
    if user_input.strip().lower() in ["quit", "exit"]:
        break

    history.append({"role": "user", "content": user_input})
    reply = chat_once(history)
    print("模型：", reply)
    history.append({"role": "assistant", "content": reply})