### STORE CHUNKS + EMBEDDINGS

import uuid # for generating unique identifiers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def store_chunks(chunks):
    # Opens local Qdrant database
    # DB stored in "qdrant_db" folder
    client = QdrantClient(path="qdrant_db")

    # collection- grp of records inside Qdrant to store the vectors
    collection_name = "research_papers"

    # checks if the collection already exists, if not it creates a new one
    try:
        # deletes existing collection and recreates an empty one before storing chunks
        # to rebuild the collection from scratch rather than adding duplicates
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print("Old collection deleted.")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3072, # number of dims in the embedding vector
                distance=Distance.COSINE # to compare embeddings
            )
        )
        print("New empty collection created.")

        # Prepare and store the chunks
        points = [] # to hold records

        # go through each chunk and assign each one an ID
        # 1 chunk = 1 Qdrant record (a point)
        for chunk in chunks:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), # unique random ID (UUID)
                    vector=chunk["embedding"], # the 3072-dim embedding vector
                    payload={ # text with metadata
                        "text": chunk["text"], # needed to retrieve the original text when searching for similar embeddings
                        "source": chunk["source"],
                        "page no": chunk["page no"]
                    }
                )
            )

        # saves record into Qdrant
        # upsert- insert new records or updates existing ones
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        print("Chunks stored:", len(points))

    finally:
        client.close() # close local DB connection