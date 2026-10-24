from google import genai
from config import GEMINI_API_KEY
from qdrant_client import QdrantClient

r"""Connect to Gemini and Qdrant"""
genai_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(path="qdrant_db")

question = "How does multi-head attention work?"

r"""Turn the question into an embedding"""
response = genai_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = response.embeddings[0].values

r"""Retrieve the 3 most relevant chunks"""
try:
    results = qdrant_client.query_points(
        collection_name="research_papers",
        query=question_embedding,
        limit=3,
        with_payload=True
    )

    # Collect the retrieved evidence
    evidence = []

    for result in results.points: # goes through each retrieved chunk
        evidence.append(
            f"Page {result.payload['page no']}:\n"
            f"{result.payload['text']}"
        )

    context = "\n\n".join(evidence)

    # Ask Gemini to answer using the retrieved evidence
    prompt = f"""
    Answer the question using only the evidence below.

    If the evidence is insufficient, say so.
    Cite the relevant page numbers in your answer.

    Evidence:
    {context}

    Question: {question}
    """

    response = genai_client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    print("\nAnswer:\n")
    print(response.text)

finally:
    qdrant_client.close()