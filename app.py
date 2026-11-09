
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import answer_question

app = FastAPI(
    title="Research Evidence Agent",
    description="A multi-paper research assistant powered by RAG.",
    version="1.0.0"
)

# pydantic model
class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Research Evidence Agent API is running!"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = answer_question(request.question)
    return result