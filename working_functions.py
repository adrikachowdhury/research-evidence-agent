def _collect_research_papers(pages, document, pdf_path):    
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
    
        if text:
            pages.append({
                "text": text,
                "page no": page_number,
                "source": pdf_path
            })
    
    document.close()

def _chunking_texts(chunk_size, overlap, chunks, page, start, text):
    end = min(start + chunk_size, len(text)) # chunk ending index
    chunk_text = text[start:end].strip() # select chars between start and end (end position excluded)
    
    if chunk_text:
        chunks.append({
            "text": chunk_text,
            "source": page["source"],
            "page no": page["page no"]
        })

    if end == len(text):
        return None
    
    # Move forward, leaving some text shared with the next chunk
    start = end - overlap
    return start

def _embed_chunks(client, chunks):
    # Generate embeddings for all chunks in the batch
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[chunk["text"] for chunk in chunks]
    )

    # Match each embedding with its original chunk
    for chunk, embedding in zip(chunks, response.embeddings):
        chunk["embedding"] = embedding.values