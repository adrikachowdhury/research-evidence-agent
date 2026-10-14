import pymupdf, re
from google import genai
from config import GEMINI_API_KEY

# open PDF
pdf_path = "data/papers/attention_is_all_you_need.pdf"
document = pymupdf.open(pdf_path)

r"""Collect text page by page"""

pages = [] # store info about every page containing text

r"""
page- current PDF page
page_number- page number of the current page

enumerate- gives item and its positions
start=1- makes page numbering start from 1
"""

for page_number, page in enumerate(document, start=1):
    text = page.get_text("text").strip() # extracts text (text mode)

    # skip pages with no text (empty pages)
    if text:
        pages.append({
            "text": text, # extracted pg text
            "page no": page_number, # page number
            "source": pdf_path # source PDF file
        })
    r"""
    page[0]- 1st page's record
    page[0]["text"]- 1st page's text
    """

# print("Pages extracted:", len(pages))
# print(pages[0]["page no"])
# print(pages[0]["text"][:500])

r"""
take one page's text
cut into smaller pieces
save each piece
and repeat until the page is finished
"""

chunk_size = 1500  # maximum target size in chars
overlap = 200  # shared chars between neighbouring chunks
chunks = []  # store the finished chunks

# remember, page is one dictionary among the list of dictionaries (pages)
for page in pages:
    text = page["text"] # extract page's text
    start = 0 # chunk starting index (char position 0)

    # loops till the very end of the page's text
    while start < len(text):
        end = min(start + chunk_size, len(text)) # chunk ending index
        chunk_text = text[start:end].strip() # select chars between start and end (end position excluded)

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page no": page["page no"]
            })

        # checks if it has reached page's end
        # after saving each chunk
        if end == len(text):
            break

        # Move forward, leaving some text shared with the next chunk
        start = end - overlap

# print("Pages extracted:", len(pages))
# print("Chunks created:", len(chunks))

# # Inspect the last chunk
# print("\nLast chunk:")
# print("Source:", chunks[37]["source"])
# print("Page:", chunks[37]["page no"])
# print("Length:", len(chunks[37]["text"]))
# print("Text:", repr(chunks[37]["text"][:500]))

# # Inspect the first chunk
# print("Page:", chunks[0]["page no"])
# print("Length:", len(chunks[0]["text"]))
# print("Text:", repr(chunks[0]["text"][:500]))

# Connect to Google's API
client = genai.Client(api_key=GEMINI_API_KEY)

# processes every chunk and generates embeddings for each chunk's text
# makes a separate API call for each chunk
for chunk in chunks:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk["text"]
    )

    chunk["embedding"] = response.embeddings[0].values # creating new key-value pair in the chunk dict

# print("Chunks with embeddings:", len(chunks))
# print("Embedding dimensions:", len(chunks[0]["embedding"]))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Create a local Qdrant database
client = QdrantClient(path="qdrant_db")

collection_name = "research_papers"

# Create a collection to store the vectors
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=3072,
            distance=Distance.COSINE
        )
    )

# Prepare and store the chunks
points = []

for chunk_id, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=chunk_id,
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page no": chunk["page no"]
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

print("Chunks stored:", len(points))