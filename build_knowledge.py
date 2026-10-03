
import os

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================
# CONFIG
# =========================================

KNOWLEDGE_DIR = "knowledge"
VECTOR_DB_DIR = "vector_db"

COLLECTION_NAME = "eaut_knowledge"

# Số ký tự tối đa trong một chunk
CHUNK_SIZE = 500

# Số ký tự chồng lên giữa hai chunk
CHUNK_OVERLAP = 100


# =========================================
# 1. CHECK KNOWLEDGE FOLDER
# =========================================

if not os.path.exists(KNOWLEDGE_DIR):
    print(f"Không tìm thấy thư mục: {KNOWLEDGE_DIR}")
    exit()


# =========================================
# 2. LOAD EMBEDDING MODEL
# =========================================

print("=========================================")
print("ĐANG TẢI EMBEDDING MODEL")
print("=========================================")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model đã tải xong.")


# =========================================
# 3. CONNECT TO CHROMA
# =========================================

print("\n=========================================")
print("ĐANG MỞ VECTOR DATABASE")
print("=========================================")

client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR
)

# Nếu collection đã tồn tại thì xóa collection cũ
# để tránh bị thêm dữ liệu trùng khi chạy lại.
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )

    print("Đã xóa knowledge base cũ.")

except Exception:
    print("Chưa có knowledge base cũ.")


collection = client.create_collection(
    name=COLLECTION_NAME
)

print("Collection đã được tạo.")


# =========================================
# 4. READ TXT FILES
# =========================================

print("\n=========================================")
print("ĐANG ĐỌC KNOWLEDGE")
print("=========================================")

documents = []
metadatas = []
ids = []

document_id = 0

txt_files = [
    filename
    for filename in os.listdir(KNOWLEDGE_DIR)
    if filename.lower().endswith(".txt")
]

if len(txt_files) == 0:
    print("Không tìm thấy file .txt trong thư mục knowledge.")
    exit()


print(f"Tìm thấy {len(txt_files)} file .txt")


for filename in txt_files:

    file_path = os.path.join(
        KNOWLEDGE_DIR,
        filename
    )

    print(f"\nĐang đọc: {filename}")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()


    if not text:
        print("  File rỗng, bỏ qua.")
        continue


    # =====================================
    # 5. SPLIT TEXT INTO CHUNKS
    # =====================================

    start = 0

    file_chunk_count = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            documents.append(chunk)

            metadatas.append({
                "source": filename
            })

            ids.append(
                f"document_{document_id}"
            )

            document_id += 1

            file_chunk_count += 1


        # Di chuyển vị trí đọc
        start += CHUNK_SIZE - CHUNK_OVERLAP


    print(
        f"  Đã tạo {file_chunk_count} chunks."
    )


# =========================================
# 6. CHECK DATA
# =========================================

print("\n=========================================")
print("THỐNG KÊ")
print("=========================================")

print(
    f"Tổng số chunks: {len(documents)}"
)

if len(documents) == 0:
    print("Không có dữ liệu để tạo embedding.")
    exit()


# =========================================
# 7. CREATE EMBEDDINGS
# =========================================

print("\n=========================================")
print("ĐANG TẠO EMBEDDINGS")
print("=========================================")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
)

print("\nĐã tạo embeddings.")


# =========================================
# 8. SAVE TO CHROMA
# =========================================

print("\n=========================================")
print("ĐANG LƯU VECTOR DATABASE")
print("=========================================")

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)


# =========================================
# 9. FINISH
# =========================================

print("\n")
print("=========================================")
print("BUILD KNOWLEDGE BASE HOÀN TẤT!")
print("=========================================")

print(f"Files: {len(txt_files)}")
print(f"Chunks: {len(documents)}")
print(f"Vector DB: {VECTOR_DB_DIR}")
print(f"Collection: {COLLECTION_NAME}")

print("=========================================")

