from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "EAUT Admission Chatbot is running!"
    }