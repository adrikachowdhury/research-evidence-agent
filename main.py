### Runs the initial paper-processing and database-building workflow (EXTRACT > CHUNK > EMBED)

import pymupdf, os, json, time
from google import genai
from config import GEMINI_API_KEY
from vector_store import store_chunks
from working_functions import (
    _collect_research_papers,
    _chunking_texts,
    _process_and_save_embeddings
)
"""Finding and reading the PDF files"""
pdf_folder = "data/papers"

pdf_files = [
    os.path.join(pdf_folder, file)
    for file in os.listdir(pdf_folder)
    if file.endswith(".pdf")
]

pages = [] # store info about every page of all papers containing text

for pdf_path in pdf_files: # visits each PDF
    document = pymupdf.open(pdf_path)
    _collect_research_papers(pages, document, pdf_path)

"""Splitting the extracted text into chunks"""
chunk_size = 1500  # maximum target size
overlap = 200  # shared chars between neighbouring chunks
chunks = []  # store the finished chunks

for page in pages:
    text = page["text"] # extract page's text
    start = 0 # chunk starting index (char position 0)

    while start < len(text):
        next_start = _chunking_texts(chunk_size, overlap, chunks, page, start, text)
        start = next_start

        # stops the loops when no page remains
        if next_start is None:
            break

"""Checkpoint file for saving completed embeddings"""
checkpoint_file = "embedding_checkpoint.json"

# Load previously saved embeddings if it exists (runs after every successful batch)

# e.g.- you embedded 60 chunks yesterday but program stopped
# when running again, we don't want those 60 chunks to be emned all over again
# checkpoint lets Py recover the embeddings it already geenrated
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r", encoding="utf-8") as file: # loads saved info
        saved_chunks = json.load(file)

    # Make sure the checkpoint belongs to the same papers and chunks
    # checks text, source, page no for each chunk
    # all()- checks whether every comparison is True
    if len(saved_chunks) == len(chunks) and all(
        current["text"] == saved["text"]
        and current["source"] == saved["source"]
        and current["page no"] == saved["page no"]
        for current, saved in zip(chunks, saved_chunks)
    ):
        for current, saved in zip(chunks, saved_chunks): # restoring saved embedding
            if "embedding" in saved:
                current["embedding"] = saved["embedding"] # copies saved embedding into current chunk

        print("Previously saved embeddings loaded.")
    else:
        print("Checkpoint does not match current papers. Starting fresh.")

"""Connect to Google's API"""
client = genai.Client(api_key=GEMINI_API_KEY)

batch_size = 2

"""Find the chunks that need embeddings"""
pending_chunks = [chunk for chunk in chunks if "embedding" not in chunk]

print("Chunks remaining to embed:", len(pending_chunks))

"""Actual embedding: Processing batches and retrying"""
_process_and_save_embeddings(pending_chunks, client, time, batch_size, checkpoint_file, chunks)

print("\nEmbedding process completed.")
print("Chunks prepared:", len(chunks))
print("First chunk embedding dimension:", len(chunks[0]["embedding"]))
print("Total chunks embedded:", sum(1 for chunk in chunks if "embedding" in chunk))

# if every chunk has an embedding, store them in the vector store
if all("embedding" in chunk for chunk in chunks):
    store_chunks(chunks)
else:
    print("Some chunks are still missing embeddings. Nothing stored.")