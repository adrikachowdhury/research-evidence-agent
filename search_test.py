from google import genai
from config import GEMINI_API_KEY
from qdrant_client import QdrantClient

# Connect to Google's API
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Open the existing local Qdrant database
qdrant_client = QdrantClient(path="qdrant_db")

# Write a question
question = "What is an attention mechanism?"

# Convert the question into an embedding
response = genai_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

# list of floats representing the meaning of the ques
question_embedding = response.embeddings[0].values # [0] since there is only one ques

try:
    # Search for the 3 closest chunks
    results = qdrant_client.query_points(
        collection_name="research_papers", # collection containing paper chunks
        query=question_embedding,
        limit=3, # return up to 3 closest matches
        with_payload=True # include the chunk metadata in the results
    )

    # Display the retrieved evidence
    for result in results.points:
        print("\nSimilarity Score:", result.score)
        print("Source:", result.payload["source"]) # identifies PDF
        print("Page:", result.payload["page no"]) 
        print("Text:", result.payload["text"][:500]) # 1st 500 chars of the matched chunk

# finally- runs whether or not an exception occurs, ensuring the DB connection is closed
finally:
    # Close the database connection
    qdrant_client.close()