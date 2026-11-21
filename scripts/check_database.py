import os
from qdrant_client import QdrantClient

client = QdrantClient(path="qdrant_db")

print("Database path:", os.path.abspath("qdrant_db"))

collection_name = "research_papers"
exists = client.collection_exists(collection_name)

print("Collection exists:", exists)

if exists:
    result = client.count(
        collection_name=collection_name,
        exact=True
    )

    print("Total records:", result.count)

    # Retrieve stored records
    records, next_page = client.scroll(
        collection_name=collection_name,
        limit=200,
        with_payload=True,
        with_vectors=False
    )

    # Count records belonging to each paper
    paper_counts = {}

    for record in records:
        source = record.payload["source"]
        paper_name = os.path.basename(source)

        paper_counts[paper_name] = paper_counts.get(paper_name, 0) + 1

    print("\nRecords per paper:")

    for paper, count in paper_counts.items():
        print(f"{paper}: {count}")

else:
    print("Database is clean. No collection exists yet.")

client.close()