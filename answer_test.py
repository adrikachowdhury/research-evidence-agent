from google import genai
from config import GEMINI_API_KEY
from qdrant_client import QdrantClient

r"""Connect to Gemini and Qdrant"""
genai_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(path="qdrant_db")

question = "What programming language was used to implement the Transformer?"

r"""Turn the question into an embedding"""
response = genai_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = response.embeddings[0].values

try:

    # Retrieve the 3 most relevant chunks
    # results holds the returned info
    results = qdrant_client.query_points(
        collection_name="research_papers",
        query=question_embedding,
        limit=3,
        with_payload=True
    )

    print("\nRetrieved pages:")

    # goes through returned chunks one by one
    # result- represents one retrieved chunk at a time
    # metadata are accessible because we stored in it Qdrant in main.py
    for result in results.points: # goes through each retrieved chunk
        print("Page:", result.payload["page no"])
        print("Text:", result.payload["text"][:200])

    # Collect the retrieved evidence
    evidence = []

    # adds each retrieved chunk to that list with the page no.
    # this loops is for the usage of Gemini
    for result in results.points:
        evidence.append(
            f"Page {result.payload['page no']}:\n"
            f"{result.payload['text']}"
        )
    # joins the 3 pcs into one long string with blank lines in between 
    context = "\n\n".join(evidence) # combine the evidence

    # give Gemini its instructions and evidence
    # instruction + context + ques
    prompt = f"""
    Answer the question using only the evidence below.

    If the evidence is insufficient, say so.
    Cite the relevant page numbers in your answer.

    Evidence:
    {context}

    Question: {question}
    """
    # ask Gemini to generate an answer
    # sends that prompt to Gemini
    # Gemini reads the ques and evidence, then generates ans
    response = genai_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    print("\nAnswer:\n")
    print(response.text) # prints Gemini's response text

# after the try block finishes even if an error occurs
finally:
    qdrant_client.close()