import json

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

def _process_and_save_embeddings(pending_chunks, client, time, batch_size, checkpoint_file, chunks):
    for i in range(0, len(pending_chunks), batch_size): # start, stop, step
        batch = pending_chunks[i:i + batch_size] # receive 2 embeddings at a time

        # Retry (3) if Gemini temporarily reaches its rate limit
        for attempt in range(3):
            try:
                _embed_chunks(client, batch) # send to Gemini for embedding and add them
                break
            # str(e) - readable error message
            except Exception as e: # while Gemini rejected a batch due to rate limit
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 2:
                    print("Rate limit reached. Waiting 1 minute before retrying...")
                    time.sleep(60)
                else:
                    raise

        # Save progress after each successful batch
        with open(checkpoint_file, "w", encoding="utf-8") as file:
            json.dump(chunks, file)

        completed = sum(1 for chunk in chunks if "embedding" in chunk)
        print(f"Embedded {completed}/{len(chunks)} chunks")