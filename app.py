### RECIEVE USER'S QUESTION THROUGH API AND RETURN ANSWER WITH SOURCES

from google.genai import errors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import answer_question

# creates API application instance
app = FastAPI(
    title="Research Evidence Agent",
    description="A multi-paper research assistant powered by RAG.",
    version="1.0.0"
)

# pydantic model (defines expected input model)
class QuestionRequest(BaseModel):
    question: str

@app.get("/") # health-check endpoint
def home():
    return {
        "message": "Research Evidence Agent API is running!"
    }

@app.post("/ask") # accepts a ques
def ask_question(request: QuestionRequest):
    try:
        result = answer_question(request.question)
        return result

    except errors.ServerError as exc:
        print(f"Gemini service error: {exc}")

        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again shortly."
        )