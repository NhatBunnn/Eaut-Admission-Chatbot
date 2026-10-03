"""
RAG Service Module for EAUT Admission Chatbot.
Loads ChromaDB vector database, multilingual embedding model, and fine-tuned Qwen2.5-1.5B model.
Provides both full generation and real-time streaming capabilities.
"""

import os
import time
import logging
import threading
from typing import Generator, List, Dict, Any, Tuple, Optional

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

logger = logging.getLogger("eaut_rag")
logging.basicConfig(level=logging.INFO)

# File paths & configs relative to project root
BASE_MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH = "outputs/qwen-eaut"
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "eaut_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


class RAGService:
    _instance: Optional["RAGService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.is_loading = False
        self.load_error = None
        self.doc_count = 0
        self.load_progress = "Khởi tạo..."

    @classmethod
    def get_instance(cls) -> "RAGService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def initialize(self):
        """Initializes all models and vector databases."""
        if self.is_loaded or self.is_loading:
            return

        with self._lock:
            self.is_loading = True
            self.load_error = None

        try:
            self.load_progress = "Đang tải mô hình nhúng (Embedding Model)..."
            logger.info(self.load_progress)
            # Nạp Embedding trên CPU để dành trọn vẹn 4GB VRAM cho Qwen
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", local_files_only=True)
            except Exception:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

            self.load_progress = "Đang kết nối cơ sở dữ liệu tri thức (ChromaDB)..."
            logger.info(self.load_progress)
            if not os.path.exists(VECTOR_DB_DIR):
                raise FileNotFoundError(f"Thư mục cơ sở dữ liệu vector '{VECTOR_DB_DIR}' không tồn tại!")

            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            self.doc_count = self.collection.count()
            logger.info(f"Đã nạp {self.doc_count} tài liệu từ ChromaDB.")

            self.load_progress = "Đang tải bộ mã hóa ngôn ngữ (Tokenizer)..."
            logger.info(self.load_progress)
            tokenizer_path = LORA_MODEL_PATH if os.path.exists(LORA_MODEL_PATH) else BASE_MODEL_PATH
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            self.load_progress = f"Đang nạp mô hình Qwen2.5 trên thiết bị {self.device.upper()}..."
            logger.info(self.load_progress)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )

            if self.device == "cuda":
                base_model = base_model.to("cuda")

            if os.path.exists(LORA_MODEL_PATH):
                self.load_progress = "Đang tích hợp trọng số thích ứng LoRA tuyển sinh EAUT..."
                logger.info(self.load_progress)
                self.model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
            else:
                self.model = base_model

            self.model.eval()

            self.is_loaded = True
            self.load_progress = "Hệ thống đã sẵn sàng phục vụ!"
            logger.info("RAG Service loaded successfully!")

        except Exception as e:
            self.load_error = str(e)
            self.load_progress = f"Lỗi khởi động: {str(e)}"
            logger.exception("Failed to initialize RAG Service")
        finally:
            self.is_loading = False

    def search_knowledge(self, question: str, top_k: int = TOP_K) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Queries ChromaDB for relevant admission context."""
        if not self.collection or not self.embedding_model:
            return [], []

        query_embedding = self.embedding_model.encode([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        documents = results["documents"][0] if results and "documents" in results and results["documents"] else []
        metadatas = results["metadatas"][0] if results and "metadatas" in results and results["metadatas"] else []

        return documents, metadatas

    def build_prompt(self, question: str, documents: List[str]) -> str:
        """Constructs prompt using EAUT admission system context."""
        context = "\n\n".join(documents) if documents else "Không có tài liệu phù hợp trong cơ sở dữ liệu."

        system_prompt = (
            "Bạn là trợ lý ảo AI tư vấn tuyển sinh chính thức của Trường Đại học Công nghệ Đông Á (EAUT).\n"
            "Nhiệm vụ của bạn:\n"
            "- Trả lời câu hỏi của thí sinh và phụ huynh một cách nhiệt tình, lịch sự, chính xác và dễ hiểu.\n"
            "- Căn cứ chủ yếu vào thông tin tuyển sinh năm 2026 được cung cấp trong phần CONTEXT.\n"
            "- Sử dụng định dạng danh sách, gạch đầu dòng rõ ràng khi trình bày các ngành học, mức điểm chuẩn hoặc bảng học phí.\n"
            "- Nếu CONTEXT không có đủ thông tin, hãy khiêm tốn nói rõ bạn chưa tìm thấy thông tin này trong tài liệu hiện tại "
            "và khuyến khích thí sinh liên hệ trực tiếp Hotline Ban tuyển sinh EAUT (024.6262.7797 - 0389.898.898) hoặc website eaut.edu.vn để được hỗ trợ cụ thể nhất.\n"
            "- Không tự sáng tác hoặc suy đoán thông tin sai lệch."
        )

        user_prompt = f"CONTEXT:\n\n{context}\n\n==============================\n\nCÂU HỎI:\n\n{question}\n\nHãy trả lời câu hỏi dựa trên CONTEXT."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def generate_answer(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Generates a complete answer synchronously."""
        start_time = time.time()
        
        if not self.is_loaded:
            return {
                "answer": "Hệ thống AI đang trong quá trình nạp mô hình tri thức tuyển sinh. Vui lòng đợi trong giây lát và thử lại!",
                "sources": [],
                "elapsed_time": 0.0,
                "status": "loading"
            }

        documents, metadatas = self.search_knowledge(question, top_k=top_k)
        text = self.build_prompt(question, documents)

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        elapsed = round(time.time() - start_time, 2)

        # Format sources nicely
        formatted_sources = []
        for doc, meta in zip(documents, metadatas):
            src_name = meta.get("source", "Tài liệu EAUT")
            # clean file name if it's a path
            src_clean = os.path.basename(src_name)
            formatted_sources.append({
                "source": src_clean,
                "preview": doc.strip()[:200] + ("..." if len(doc.strip()) > 200 else "")
            })

        return {
            "answer": answer,
            "sources": formatted_sources,
            "elapsed_time": elapsed,
            "status": "success"
        }

    def generate_stream(self, question: str, top_k: int = TOP_K) -> Generator[Dict[str, Any], None, None]:
        """Streams tokens in real-time using TextIteratorStreamer."""
        if not self.is_loaded:
            yield {
                "type": "error",
                "content": "Hệ thống AI đang khởi động. Vui lòng thử lại sau vài giây!"
            }
            return

        documents, metadatas = self.search_knowledge(question, top_k=top_k)
        
        # Format sources to send at start or end
        formatted_sources = []
        for doc, meta in zip(documents, metadatas):
            src_clean = os.path.basename(meta.get("source", "Tài liệu EAUT"))
            formatted_sources.append({
                "source": src_clean,
                "preview": doc.strip()[:200] + ("..." if len(doc.strip()) > 200 else "")
            })

        # Yield sources info first
        yield {
            "type": "meta",
            "sources": formatted_sources
        }

        text = self.build_prompt(question, documents)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=400,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text:
                yield {
                    "type": "token",
                    "content": new_text
                }

        thread.join()
        yield {
            "type": "done"
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns health and status metrics of the RAG engine."""
        return {
            "is_loaded": self.is_loaded,
            "is_loading": self.is_loading,
            "progress": self.load_progress,
            "error": self.load_error,
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
            "doc_count": self.doc_count,
            "model_name": "Qwen2.5-1.5B-Instruct + LoRA EAUT",
            "embedding_name": "paraphrase-multilingual-MiniLM-L12-v2"
        }
