import torch
import chromadb

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================================
# CONFIG
# =========================================

BASE_MODEL = "models/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "outputs/qwen-eaut"

VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "eaut_knowledge"

TOP_K = 3


# =========================================
# 1. LOAD EMBEDDING MODEL
# =========================================

print("=========================================")
print("ĐANG TẢI EMBEDDING MODEL")
print("=========================================")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model OK.")


# =========================================
# 2. LOAD CHROMA DATABASE
# =========================================

print("\n=========================================")
print("ĐANG MỞ CHROMA DATABASE")
print("=========================================")

client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB OK.")
print("Số documents:", collection.count())


# =========================================
# 3. LOAD TOKENIZER
# =========================================

print("\n=========================================")
print("ĐANG TẢI TOKENIZER")
print("=========================================")

tokenizer = AutoTokenizer.from_pretrained(
    LORA_MODEL
)

print("Tokenizer OK.")


# =========================================
# 4. LOAD BASE MODEL
# =========================================

print("\n=========================================")
print("ĐANG TẢI QWEN")
print("=========================================")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
)

model = model.to("cuda")


# =========================================
# 5. LOAD LORA
# =========================================

print("\n=========================================")
print("ĐANG TẢI LORA")
print("=========================================")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL
)

model.eval()

print("Qwen + LoRA OK.")
print("Model đang chạy trên:", next(model.parameters()).device)


# =========================================
# 6. SEARCH KNOWLEDGE
# =========================================

def search_knowledge(question):

    query_embedding = embedding_model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# =========================================
# 7. GENERATE ANSWER
# =========================================

def generate_answer(question, documents):

    context = "\n\n".join(
        documents
    )

    system_prompt = """
Bạn là chatbot tư vấn tuyển sinh của Trường Đại học Công nghệ Đông Á (EAUT).

Nhiệm vụ:
- Trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.
- Chỉ sử dụng thông tin trong phần CONTEXT.
- Nếu CONTEXT không có thông tin để trả lời, hãy nói rằng bạn chưa tìm thấy thông tin phù hợp.
- Không tự bịa thông tin.
- Trả lời bằng tiếng Việt.
- Trả lời ngắn gọn, dễ hiểu.
"""

    user_prompt = f"""
CONTEXT:

{context}

==============================

CÂU HỎI:

{question}

Hãy trả lời câu hỏi dựa trên CONTEXT.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()


# =========================================
# 8. CHAT
# =========================================

print("\n")
print("=========================================")
print("EAUT ADMISSION CHATBOT")
print("=========================================")
print("Nhập 'exit' để thoát.")
print("=========================================")


while True:

    question = input("\nBạn: ")

    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:
        break

    if not question.strip():
        continue


    # -------------------------------------
    # SEARCH
    # -------------------------------------

    documents, metadatas = search_knowledge(
        question
    )


    # -------------------------------------
    # GENERATE
    # -------------------------------------

    answer = generate_answer(
        question,
        documents
    )


    print("\nBot:", answer)


    # -------------------------------------
    # SHOW SOURCES
    # -------------------------------------

    print("\nNguồn:")

    for metadata in metadatas:

        print(
            "-",
            metadata["source"]
        )