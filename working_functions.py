def _collect_pages(pages, pdf_path, page, page_number):
    text = page.get_text("text").strip() # extracts text ("text" mode)
    # skip pages with no text (empty pages)
    if text:
        pages.append({
            "text": text,
            "page no": page_number,
            "source": pdf_path
        })
    
    # page[0]- 1st page's record
    # page[0]["text"]- 1st page's text

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

def _embed_chunks(client, chunk):
    # processes every chunk and generates embeddings for each chunk's text
    # makes a separate API call for each chunk
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk["text"]
    )
    chunk["embedding"] = response.embeddings[0].values # creating new key-value pair in the chunk dict

    client.close()