import pymupdf
from google import genai
from config import GEMINI_API_KEY
from vector_store import store_chunks
from working_functions import _chunking_texts, _collect_pages, _embed_chunks

pdf_path = "data/papers/attention_is_all_you_need.pdf"
document = pymupdf.open(pdf_path)

# extract pages
pages = [] # store info about every page containing text
for page_number, page in enumerate(document, start=1):
    _collect_pages(pages, pdf_path, page, page_number)

# extract chunks
chunk_size = 1500  # maximum target size in chars
overlap = 200  # shared chars between neighbouring chunks
chunks = []  # store the finished chunks
for page in pages:
    text = page["text"] # extract page's text
    start = 0 # chunk starting index (char position 0)

    # loops till the very end of the page's text
    # until function returns None
    while start < len(text):
        next_start = _chunking_texts(chunk_size, overlap, chunks, page, start, text)

        # stops the loops when no page remains
        if next_start is None:
            break

        start = next_start

# Connect to Google's API using the API key
client = genai.Client(api_key=GEMINI_API_KEY)

for chunk in chunks:
    _embed_chunks(client, chunk) # embed each chunk one by one

store_chunks(chunks) # store embedded chunks in Qdrant