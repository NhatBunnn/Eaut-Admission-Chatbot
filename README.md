# 🎓 EAUT Admission Chatbot — Trợ Lý Tuyển Sinh AI

Hệ thống Chatbot AI thông minh tư vấn tuyển sinh dành cho **Trường Đại học Công nghệ Đông Á (EAUT)**, kết hợp kỹ thuật **RAG (Retrieval-Augmented Generation)** và mô hình ngôn ngữ lớn **Qwen2.5-1.5B-Instruct** được tinh chỉnh chuyên sâu (**QLoRA fine-tuning**).

---

## 🌟 Tính Năng Nổi Bật

- 🔍 **RAG Đa Tầng (Retrieval-Augmented Generation):**
  - Cơ sở dữ liệu vector **ChromaDB** lập chỉ mục toàn bộ cẩm nang tuyển sinh EAUT (ngành đào tạo, học phí, điểm chuẩn, hồ sơ xét tuyển, học bổng).
  - Mô hình nhúng đa ngôn ngữ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` giúp tìm kiếm ngữ nghĩa chính xác theo ngữ cảnh câu hỏi của thí sinh.
- 🧠 **Mô Hình Ngôn Ngữ Tinh Chỉnh (Fine-tuned LLM):**
  - Huấn luyện trên nền tảng **Qwen2.5-1.5B-Instruct** bằng phương pháp **QLoRA** (4-bit quantization, LoRA rank=8, alpha=16) với tập dữ liệu hỏi đáp tuyển sinh thực tế.
  - Phản hồi bằng tiếng Việt tự nhiên, chuẩn xác thông tin trường EAUT và hạn chế tối đa hiện tượng "ảo giác" (hallucination).
- ⚡ **Giao Diện Web Hiện Đại & Real-time Streaming:**
  - Tích hợp máy chủ **FastAPI** và giao diện người dùng Web tương tác trực quan.
  - Hỗ trợ trả lời dạng dòng truyền thời gian thực (**Server-Sent Events - SSE**) tạo trải nghiệm mượt mà như ChatGPT.
- 💻 **Tối Ưu Hóa Tài Nguyên Phần Cứng:**
  - Thiết kế tối ưu cho máy tính cá nhân và laptop phổ thông, chạy ổn định trên card đồ họa chỉ từ **4GB VRAM** (ví dụ: NVIDIA GeForce RTX 3050 Laptop GPU).

---

## 🏗️ Kiến Trúc Hệ Thống

```
               [ Thí sinh / Người dùng ]
                           │
                           ▼
              ┌─────────────────────────┐
              │   FastAPI Web Server    │
              │  (Giao diện / REST API) │
              └────────────┬────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [ Trích xuất ngữ cảnh ]    [ Sinh câu trả lời ]
     MiniLM Embeddings           Qwen2.5-1.5B + LoRA
             │                           │
             ▼                           │
     [ ChromaDB Vector ] ───(Context)───►┘
   (184 chunks cẩm nang)
```

---

## 📂 Cấu Trúc Dự Án

```plaintext
Eaut-Admission-Chatbot/
├── app/                        # Ứng dụng Web & API
│   ├── main.py                 # Máy chủ FastAPI (endpoints, streaming, handbook)
│   ├── rag_service.py          # Module dịch vụ RAG & quản lý nạp mô hình
│   └── static/                 # Giao diện Web tĩnh (HTML, CSS, JS, hình ảnh)
│       └── index.html          # Trang chủ Chatbot EAUT
├── data/                       # Dữ liệu huấn luyện hỏi-đáp tuyển sinh
│   ├── train.jsonl             # Tập huấn luyện SFT
│   ├── validation.jsonl        # Tập đánh giá
│   └── test.jsonl              # Tập kiểm thử
├── knowledge/                  # Dữ liệu thô cẩm nang tuyển sinh EAUT
│   ├── diem_chuan.txt          # Điểm trúng tuyển các năm
│   ├── ho_so.txt               # Hướng dẫn hồ sơ & quy trình
│   ├── hoc_bong.txt            # Chính sách học bổng
│   ├── hoc_phi.txt             # Biểu mức học phí & hỗ trợ
│   ├── nganh_dao_tao.txt       # Danh mục các ngành đào tạo
│   ├── phuong_thuc_xet_tuyen.txt # Các phương thức xét tuyển
│   └── thoi_gian_tuyen_sinh.txt# Lịch trình mốc thời gian
├── models/                     # Thư mục chứa trọng số mô hình gốc
│   └── Qwen2.5-1.5B-Instruct/  # Checkpoint mô hình gốc Qwen 1.5B
├── outputs/                    # Kết quả sau huấn luyện
│   └── qwen-eaut/              # Trọng số thích ứng LoRA (adapter PEFT)
├── vector_db/                  # Cơ sở dữ liệu ChromaDB lưu chỉ mục vector
├── build_knowledge.py          # Script phân đoạn tài liệu & tạo vector DB
├── query_knowledge.py          # Script kiểm tra truy vấn vector DB qua CLI
├── train.py                    # Script huấn luyện QLoRA cho Qwen
├── test_model.py               # Script test mô hình base
├── test_finetuned.py           # Script test mô hình tinh chỉnh LoRA
├── rag.py                      # Chatbot RAG chạy trên Terminal (CLI)
├── run.bat                     # File kích hoạt phần mềm nhanh 1-click (Windows)
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Yêu Cầu Hệ Thống

- **Hệ điều hành:** Windows 10/11 (64-bit), Linux hoặc macOS.
- **Python:** 3.10 hoặc 3.11.
- **Phần cứng đề xuất:**
  - **GPU:** NVIDIA có hỗ trợ CUDA (tối thiểu 4GB VRAM, ví dụ: GTX 1650 Ti, RTX 3050, RTX 4060,...).
  - **RAM:** Tối thiểu 8GB (khuyên dùng 16GB).
  - **Ổ cứng:** Trống tối thiểu 10GB.

---

## 🚀 Hướng Dẫn Khởi Chạy

### Cách 1: Chạy nhanh bằng file `run.bat` (Khuyên dùng trên Windows)
1. Mở thư mục dự án trong File Explorer.
2. Nhấp đúp chuột vào file **`run.bat`**.
3. Cửa sổ dòng lệnh sẽ tự động kiểm tra cổng mạng, kích hoạt môi trường ảo và khởi động máy chủ.
4. Mở trình duyệt web và truy cập: **[http://localhost:8000](http://localhost:8000)**.

---

### Cách 2: Khởi chạy thủ công qua Terminal (PowerShell / CMD)

1. **Mở Terminal tại thư mục gốc của dự án.**
2. **Kích hoạt môi trường ảo:**
   ```powershell
   # Trên Windows PowerShell:
   .\venv\Scripts\activate

   # Hoặc trên Command Prompt (cmd):
   call venv\Scripts\activate.bat
   ```
3. **Khởi động server bằng Uvicorn:**
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
4. **Truy cập ứng dụng:**
   - Giao diện người dùng: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - Tài liệu API tương tác (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔄 Quy Trình Xây Dựng & Huấn Luyện (Pipeline)

Nếu bạn muốn cập nhật dữ liệu tuyển sinh mới hoặc huấn luyện lại mô hình:

### 1. Cập nhật dữ liệu & Tạo lại Vector DB
Chỉnh sửa các file văn bản trong thư mục `knowledge/`, sau đó chạy:
```powershell
python build_knowledge.py
```
Dữ liệu sẽ được chia nhỏ (chunking), gắn nhãn metadata và lưu trữ trong thư mục `vector_db/`.

### 2. Huấn luyện mô hình (QLoRA Fine-tuning)
Chuẩn bị tập dữ liệu hỏi đáp chuẩn dạng JSON Lines trong thư mục `data/` (`train.jsonl`, `validation.jsonl`), sau đó tiến hành train:
```powershell
python train.py
```
Quá trình huấn luyện sử dụng `SFTTrainer` (thư viện `trl`), lưu trọng số LoRA thích ứng vào `outputs/qwen-eaut/`.

### 3. Kiểm thử qua dòng lệnh (CLI)
- Kiểm tra truy xuất tri thức:
  ```powershell
  python query_knowledge.py
  ```
- Trò chuyện với RAG Chatbot trực tiếp trên Terminal:
  ```powershell
  python rag.py
  ```

---

## 📡 Danh Sách API Endpoints

| Phương thức | Đường dẫn | Chức năng |
| :--- | :--- | :--- |
| `GET` | `/` | Trả về giao diện người dùng Web Chatbot |
| `GET` | `/api/status` | Kiểm tra trạng thái nạp mô hình, thiết bị (CUDA/CPU), số lượng tài liệu |
| `POST` | `/api/chat` | Gửi câu hỏi tuyển sinh, trả về toàn bộ câu trả lời kèm tài liệu tham khảo |
| `POST` | `/api/chat/stream` | Gửi câu hỏi, stream từng token thời gian thực dạng SSE (`text/event-stream`) |
| `GET` | `/api/handbook` | Lấy dữ liệu cẩm nang tuyển sinh tổng hợp (phương thức, mốc thời gian, học bổng) |
| `POST` | `/api/feedback` | Ghi nhận đánh giá phản hồi (Like/Dislike) từ người dùng |
| `GET` | `/docs` | Tài liệu API Swagger UI tương tác trực tiếp |

---

## 🏫 Thông Tin Trường Đại Học Công Nghệ Đông Á (EAUT)
- **Tên trường:** Trường Đại học Công nghệ Đông Á (East Asia University of Technology)
- **Mã trường:** `DDA`
- **Địa chỉ:** Đường Trịnh Văn Bô, Phường Xuân Phương, Quận Nam Từ Liêm, TP. Hà Nội
- **Website chính thức:** [https://eaut.edu.vn](https://eaut.edu.vn)
- **Cổng thông tin tuyển sinh:** [https://tuyensinh.eaut.edu.vn](https://tuyensinh.eaut.edu.vn)
