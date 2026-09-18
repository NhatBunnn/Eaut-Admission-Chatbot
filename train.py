import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =========================
# CONFIG
# =========================

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"

OUTPUT_DIR = "outputs/qwen-eaut"


# =========================
# 1. LOAD DATASET
# =========================

print("Đang tải dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))


# =========================
# 2. LOAD TOKENIZER
# =========================

print("\nĐang tải tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# 3. LOAD MODEL 4-BIT
# =========================

print("\nĐang tải model 4-bit...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
)

model.config.use_cache = False


# =========================
# 4. FORMAT DATA
# =========================

def format_example(example):
    messages = [
        {
            "role": "user",
            "content": example["question"],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# =========================
# 5. LORA
# =========================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# =========================
# 6. TRAINING CONFIG
# =========================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    # Sequence length
    max_length=512,

    # Logging
    logging_steps=10,

    # Evaluation
    eval_strategy="steps",
    eval_steps=100,

    # Save checkpoint
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    # GPU
    fp16=True,

    # Memory optimization
    gradient_checkpointing=True,

    # 8-bit optimizer
    optim="paged_adamw_8bit",

    # Không dùng WandB
    report_to="none",
)


# =========================
# 7. CREATE TRAINER
# =========================

print("\nĐang tạo Trainer...")

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    peft_config=peft_config,

    formatting_func=format_example,

    processing_class=tokenizer,
)


# =========================
# 8. START TRAINING
# =========================

print("\n")
print("==============================")
print("BẮT ĐẦU TRAIN")
print("==============================")
print("\n")

trainer.train()


# =========================
# 9. SAVE MODEL
# =========================

print("\nĐang lưu model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


print("\n")
print("==============================")
print("TRAIN HOÀN TẤT!")
print("==============================")
print("Model được lưu tại:")
print(OUTPUT_DIR)
print("==============================")