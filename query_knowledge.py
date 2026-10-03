import chromadb
from sentence_transformers import SentenceTransformer


VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "eaut_knowledge"


print("Đang tải embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model OK.")


print("\nĐang mở ChromaDB...")

client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB OK.")
print("Số documents:", collection.count())


while True:

    question = input("\nBạn: ")

    if question.lower() in ["exit", "quit", "q"]:
        break

    query_embedding = embedding_model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    print("\n========== KẾT QUẢ ==========")

    for i in range(len(results["documents"][0])):

        document = results["documents"][0][i]
        source = results["metadatas"][0][i]

        print(f"\n--- Kết quả {i + 1} ---")
        print("Nguồn:", source["source"])
        print("Nội dung:")
        print(document)

    print("\n==============================")

