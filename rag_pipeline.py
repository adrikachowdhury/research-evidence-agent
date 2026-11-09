from google import genai
from config import GEMINI_API_KEY
from qdrant_client import QdrantClient
import os

def retrieve_evidence(question, top_k=3):
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    qdrant_client = QdrantClient(path="qdrant_db")

    try:
        # Convert the question into an embedding
        response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )

        question_embedding = response.embeddings[0].values

        # Retrieve the most relevant chunks
        results = qdrant_client.query_points(
            collection_name="research_papers",
            query=question_embedding,
            limit=top_k,
            with_payload=True
        )

        evidence = []
        sources = []

        for result in results.points:
            source = os.path.basename(result.payload["source"])
            page = result.payload["page no"]
            text = result.payload["text"]

            evidence.append(
                f"Source: {source}, Page {page}:\n{text}"
            )

            sources.append({
                "source": source,
                "page": page
            })

        return "\n\n".join(evidence), sources

    finally:
        qdrant_client.close()

def generate_answer(question, context):
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    Answer the question using only the evidence below.

    Rules:
    - Make only claims supported by the evidence.
    - If the evidence is insufficient, clearly say so.
    - Cite the source and page number for factual claims.
    - Do not invent facts or citations.

    Evidence:
    {context}

    Question: {question}
    """

    response = genai_client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return response.text

def answer_question(question):
    context, sources = retrieve_evidence(question)

    if not context:
        return {
            "answer": "I couldn't find relevant evidence in the stored research papers.",
            "sources": []
        }

    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": sources
    }