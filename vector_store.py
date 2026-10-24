from qdrant_client import QdrantClient, qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# Create a local database
# Opens in "qdrant_db" folder
client = QdrantClient(path="qdrant_db")

# create a collection to store the vectors
# collection- like a table/container in a DB
collection_name = "research_papers"

# Create a collection to store the vectors
#  checks if the collection already exists, if not it creates a new one
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=3072, # size of the embedding vector (3072 dims for Gemini embeddings)
            distance=Distance.COSINE # compare using cosine similarity
        )
    )

# Prepare and store the chunks
# converts each chunk into a format Qdrant understands
points = [] # to hold records

# go through each chunk and assign each one an ID
for chunk_id, chunk in enumerate(chunks):
    points.append(
        # packages the ID, vector, and metadata into one Qdrant record (each chunk)
        PointStruct(
            id=chunk_id, # gives each record a unique ID
            vector=chunk["embedding"], # the 3072-dim embedding vector
            payload={
                "text": chunk["text"], #  original chunk text
                "source": chunk["source"],
                "page no": chunk["page no"]
            }
        )
    )

# save in the Qdrant collection
# upsert- adds new records or updates existing ones
client.upsert(
    collection_name=collection_name,
    points=points
)

print("Chunks stored:", len(points))

# close local DB connection
qdrant_client.close()