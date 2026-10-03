@echo off
cd /d "%~dp0"
title EAUT Admission Chatbot
set PYTHONIOENCODING=utf-8

echo ===================================================
echo     HE THONG TRO LY TUYEN SINH AI - EAUT CHATBOT
echo ===================================================

for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

echo Dang khoi dong may chu tai http://127.0.0.1:8000 ...
echo Vui long doi trong giay lat...
call ".\venv\Scripts\activate.bat"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo May chu da dung lai.
pause
