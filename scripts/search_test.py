from google import genai
from config import GEMINI_API_KEY
from qdrant_client import QdrantClient

# Connect to Google's API
genai_client = genai.Client(api_key=GEMINI_API_KEY)

qdrant_client = QdrantClient(path="qdrant_db")
question = "Why does the Transformer use attention?"

response = genai_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = response.embeddings[0].values # [0] since there is only one ques

try:
    results = qdrant_client.query_points(
        collection_name="research_papers",
        query=question_embedding,
        limit=3,
        with_payload=True
    )

    # Display the retrieved evidence
    for result in results.points:
        print("\nSimilarity Score:", result.score)
        print("Source:", result.payload["source"])
        print("Page:", result.payload["page no"]) 
        print("Text:", result.payload["text"][:500]) # 1st 500 chars of the matched chunk

# finally- runs whether or not an exception occurs, ensuring the DB connection is closed
finally:
    qdrant_client.close()