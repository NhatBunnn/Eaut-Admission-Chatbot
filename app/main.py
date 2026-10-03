"""
FastAPI Server for EAUT Admission Chatbot.
Provides REST and SSE endpoints for admissions inquiry, system health, and handbook data.
"""

import os
import json
import asyncio
import threading
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_service import RAGService


# Request Models
class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 3


class FeedbackRequest(BaseModel):
    message_id: str
    rating: str  # "like" or "dislike"
    comment: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG models in a background worker thread so the server opens ports immediately
    rag = RAGService.get_instance()
    init_thread = threading.Thread(target=rag.initialize, daemon=True)
    init_thread.start()
    yield


app = FastAPI(
    title="EAUT Admission Chatbot API",
    description="Hệ thống Trợ lý Tuyển sinh AI - Trường Đại học Công nghệ Đông Á (EAUT)",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>EAUT Admission Chatbot UI is initializing...</h1>")


@app.get("/api/status")
async def get_status():
    rag = RAGService.get_instance()
    return rag.get_status()


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    rag = RAGService.get_instance()
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống!")

    # Run blocking generation in asyncio thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, rag.generate_answer, query, request.top_k or 3)
    return result


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    rag = RAGService.get_instance()
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống!")

    async def event_generator():
        # Iterate over generator produced by RAGService
        for event in rag.generate_stream(query, top_k=request.top_k or 3):
            data_str = json.dumps(event, ensure_ascii=False)
            yield f"data: {data_str}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/sample-questions")
async def get_sample_questions():
    """Curated questions for student quick interaction."""
    return [
        {
            "category": "Ngành đào tạo & Điểm chuẩn",
            "icon": "🎓",
            "questions": [
                "Trường EAUT đào tạo những ngành học nào năm 2026?",
                "Điểm chuẩn ngành Công nghệ thông tin năm nay dự kiến bao nhiêu?",
                "Chuyên ngành Trí tuệ nhân tạo ứng dụng học những gì?",
                "Mã ngành Công nghệ kỹ thuật Ô tô của trường là gì?"
            ]
        },
        {
            "category": "Học phí & Học bổng",
            "icon": "💰",
            "questions": [
                "Học phí 1 tín chỉ và 1 kỳ của trường là bao nhiêu?",
                "Chính sách học bổng dành cho tân sinh viên năm 2026 như thế nào?",
                "Trường có cam kết không tăng học phí đột ngột trong suốt khóa học không?",
                "Học sinh giỏi cấp tỉnh có được xét cấp học bổng không?"
            ]
        },
        {
            "category": "Phương thức xét tuyển & Hồ sơ",
            "icon": "📑",
            "questions": [
                "EAUT có những phương thức xét tuyển nào?",
                "Điều kiện xét tuyển bằng học bạ THPT năm 2026 là gì?",
                "Hồ sơ đăng ký xét tuyển trực tuyến cần chuẩn bị những giấy tờ gì?",
                "Trường có xét điểm thi Đánh giá năng lực của ĐHQG không?"
            ]
        },
        {
            "category": "Thời gian & Thủ tục",
            "icon": "⏰",
            "questions": [
                "Thời gian đăng ký nguyện vọng đại học năm 2026 là khi nào?",
                "Bao giờ EAUT công bố kết quả trúng tuyển đợt 1?",
                "Thời hạn cuối cùng để xác nhận nhập học trực tuyến là ngày nào?",
                "Trường Đại học Công nghệ Đông Á có xét tuyển bổ sung không?"
            ]
        }
    ]


@app.get("/api/handbook")
async def get_admission_handbook():
    """Fast structured handbook data for interactive modal popup."""
    return {
        "university": {
            "name": "Trường Đại học Công nghệ Đông Á",
            "code": "DDA",
            "slogan": "Tri thức - Trực quan - Thực tiễn",
            "hotline": "024.6262.7797 - 0389.898.898",
            "email": "tuyensinh@eaut.edu.vn",
            "website": "https://eaut.edu.vn",
            "address": "Đường Trịnh Văn Bô, Phường Xuân Phương, Quận Nam Từ Liêm, Hà Nội"
        },
        "methods": [
            {
                "title": "Phương thức 1: Xét tuyển học bạ THPT",
                "desc": "Xét kết quả học tập THPT lớp 11 và học kỳ 1 lớp 12 hoặc cả năm lớp 12 theo tổ hợp 3 môn xét tuyển. Tổng điểm đạt từ 18.0 trở lên."
            },
            {
                "title": "Phương thức 2: Xét điểm thi tốt nghiệp THPT 2026",
                "desc": "Căn cứ vào điểm thi tốt nghiệp THPT theo các tổ hợp môn tương ứng của từng ngành."
            },
            {
                "title": "Phương thức 3: Xét kết quả thi Đánh giá năng lực / Đánh giá tư duy",
                "desc": "Sử dụng kết quả kỳ thi ĐGNL của ĐHQG Hà Nội hoặc kỳ thi ĐGTD của ĐH Bách Khoa Hà Nội."
            },
            {
                "title": "Phương thức 4: Xét tuyển thẳng & Ưu tiên xét tuyển",
                "desc": "Theo quy định hiện hành của Bộ Giáo dục & Đào tạo và quy chế tuyển sinh của EAUT."
            }
        ],
        "timeline_2026": [
            {"milestone": "02/7 - 14/7/2026", "event": "Thí sinh đăng ký và điều chỉnh nguyện vọng xét tuyển trên hệ thống chung của Bộ GD&ĐT (Mã trường: DDA)."},
            {"milestone": "15/7 - 21/7/2026 (17h00)", "event": "Nộp lệ phí xét tuyển trực tuyến theo số lượng nguyện vọng."},
            {"milestone": "Trước 17h00 ngày 13/8/2026", "event": "EAUT và các trường đại học công bố điểm chuẩn và danh sách trúng tuyển đợt 1."},
            {"milestone": "Trước 17h00 ngày 21/8/2026", "event": "Thí sinh hoàn thành xác nhận nhập học trực tuyến trên cổng của Bộ GD&ĐT."},
            {"milestone": "Từ 22/8/2026", "event": "Bắt đầu xét tuyển các đợt bổ sung (nếu còn chỉ tiêu theo từng ngành cụ thể)."}
        ],
        "scholarships": [
            {"tier": "Học bổng Kim Cương (100%)", "benefit": "Miễn 100% học phí toàn khóa học dành cho thí sinh xuất sắc đạt giải quốc gia hoặc điểm thi THPT cực cao."},
            {"tier": "Học bổng Vàng (50%)", "benefit": "Giảm 50% học phí năm thứ nhất cho học sinh giỏi trường chuyên, học sinh đạt thành tích tiêu biểu."},
            {"tier": "Học bổng Đồng hành EAUT", "benefit": "Nhiều suất học bổng từ 3.000.000đ - 10.000.000đ do các tập đoàn doanh nghiệp đối tác tài trợ cho tân sinh viên."},
            {"tier": "Chính sách hỗ trợ ký túc xá", "benefit": "Ưu tiên chỗ ở ký túc xá khang trang, hiện đại cho sinh viên ngoại tỉnh và đối tượng chính sách."}
        ]
    }