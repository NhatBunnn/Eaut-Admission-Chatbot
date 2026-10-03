import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# =========================
# CONFIG
# =========================

BASE_MODEL = "models/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "outputs/qwen-eaut"


# =========================
# 1. LOAD TOKENIZER
# =========================

print("Đang tải tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    LORA_MODEL
)


# =========================
# 2. LOAD BASE MODEL
# =========================

print("Đang tải base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
)

model = model.to("cuda")


# =========================
# 3. LOAD LORA
# =========================

print("Đang tải LoRA...")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL,
)

model.eval()

print("Model đang chạy trên:", next(model.parameters()).device)


# =========================
# 4. TEST LOOP
# =========================

while True:

    question = input("\nBạn: ")

    if question.lower() in ["exit", "quit", "q"]:
        break

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")


    # =========================
    # GENERATE
    # =========================

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )


    # =========================
    # DECODE
    # =========================

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("\nBot:", answer)