import uuid # for generating unique identifiers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def store_chunks(chunks):
    # Create a local database
    # Opens in "qdrant_db" folder
    client = QdrantClient(path="qdrant_db")

    # create a collection to store the vectors
    # collection- like a table/container in a DB
    collection_name = "research_papers"

    # Create a collection to store the vectors
    # checks if the collection already exists, if not it creates a new one
    try:
        # deletes existing collection and recreates an empty one before storing chunks
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print("Old collection deleted.")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE
            )
        )
        print("New empty collection created.")

        # Prepare and store the chunks
        # converts each chunk into a format Qdrant understands
        points = [] # to hold records

        # go through each chunk and assign each one an ID
        for chunk in chunks:
            points.append(
                # packages the ID, vector, and metadata into one Qdrant record (each chunk)
                PointStruct(
                    id=str(uuid.uuid4()), # gives each record a unique random UUID
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

    finally:
        client.close() # close local DB connection