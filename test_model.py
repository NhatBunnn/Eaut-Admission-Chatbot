import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"

print("Đang tải tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Đang tải model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
)

model = model.to("cuda")

print("Model đã chạy trên:", next(model.parameters()).device)

messages = [
    {
        "role": "user",
        "content": "Ngành Công nghệ thông tin có cơ hội thăng tiến không?"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nCâu trả lời:")
print(answer)