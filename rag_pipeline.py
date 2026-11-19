### Coordinates the RAG process: retrieval → answer generation

from google import genai
from google.genai import types
from config import GEMINI_API_KEY
from qdrant_client import QdrantClient
import os, time

"""RETRIEVAL"""
# find relevant passages
# top_k = 3: retrieve the 3 most relevant chunks
def retrieve_evidence(question, top_k=3):
    genai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=1.0,
            max_delay=2.0
        )
    )
)
    qdrant_client = QdrantClient(path="qdrant_db") # where the chunks are stored

    try:
        start_time = time.time()

        response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )

        print(f"Embedding time: {time.time() - start_time:.2f} seconds")

        question_embedding = response.embeddings[0].values

        # Retrieve the most relevant chunks from Qdrant
        results = qdrant_client.query_points(
            collection_name="research_papers",
            query=question_embedding,
            limit=top_k,
            with_payload=True # text + metadata
        )

        evidence = []
        sources = []

        for result in results.points:
            source = os.path.basename(result.payload["source"]) # takes only the pdf name
            page = result.payload["page no"]
            text = result.payload["text"]

            evidence.append(
                f"Source: {source}, Page {page}:\n{text}"
            )

            sources.append({
                "source": source,
                "page": page
            })
        # returns 'context'
        return "\n\n".join(evidence), sources # combines passages with blank lines in between

    finally:
        qdrant_client.close()

"""GENERATION"""
def generate_answer(question, context):
    genai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=1.0,
            max_delay=2.0
        )
    )
)

    prompt = f"""
    Answer the question using only the evidence given below.

    Rules to follow while answering:
    - Make only claims supported by the given evidence.
    - If the evidence is insufficient, clearly say so.
    - Cite the source and page number for factual claims.
    - Do not fabricate important facts or citations.

    Evidence:
    {context}

    Question:
    {question}
    """
    start_time = time.time()

    # send prompt to Gemini API
    response = genai_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    print(f"Answer generation time: {time.time() - start_time:.2f} seconds")

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
        "answer": answer, # generated answer based on the retrieved evidence
        "sources": sources
    }