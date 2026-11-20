# Research Evidence Agent: Project Guide

This document provides a closer look at how the Research Evidence Agent was implemented, how its main components work together, what was tested, and what could be improved.

It is intended to complement the main [README](README.md), not replace it.

## 1. Project Overview

The Research Evidence Agent is a small Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from a collection of research papers.

The initial document collection contains **two papers**. The goal was to build a working application that could process multiple research papers, retrieve relevant passages and generate answers with source references.

This is an application built using existing AI models and tools. It is not a newly trained or fine-tuned language model.

## 2. System Architecture

The project has two main workflows.

### A. Document Processing

The initial workflow is handled by `main.py`.

```text
Research PDFs
     |
     v
Text Extraction
   (PyMuPDF)
     |
     v
Text Chunking
     |
     v
Embedding Generation
     |
     v
Qdrant Vector Database
```

**Step 1: Extracting PDF text**

`main.py` looks inside `data/papers/` and collects files ending in `.pdf`.

Each PDF is opened using PyMuPDF. The extracted text is organised with page and source information so that the application can later identify where a retrieved passage came from.

The `_collect_research_papers()` function in `working_functions.py` handles the extraction.

**Step 2: Splitting text into chunks**

The extracted text is divided into smaller sections.

Current settings:

| Setting | Value |
|---|---:|
| Chunk size | 1,500 characters |
| Overlap | 200 characters |

The overlap means neighbouring chunks share some text. This helps preserve context when an important sentence or idea falls near a chunk boundary.

The `_chunking_texts()` function handles this process.

**Step 3: Generating embeddings**

Each chunk is converted into a numerical representation called an embedding.

The project uses Google's `gemini-embedding-001` model, which produces embeddings with 3,072 dimensions.

The `_process_and_save_embeddings()` function handles embedding generation in batches of two.

**Step 4: Saving embeddings**

Once all chunks have embeddings, `store_chunks()` sends them to the local Qdrant vector database.

Each stored record includes its vector and associated information, such as the original text, paper name and page number.

### B. Question Answering

The question-answering workflow is handled by `rag_pipeline.py` and exposed through `app.py`.

```text
User Question
      |
      v
Question Embedding
      |
      v
Qdrant Similarity Search
      |
      v
Top 3 Retrieved Chunks
      |
      v
Gemini Answer Generation
      |
      v
Answer + Source References
```

1. The user's question is converted into an embedding using the same embedding model.
2. Qdrant searches for the three most relevant stored chunks.
3. The retrieved text is passed to the answer-generation model along with the question.
4. Gemini generates an answer using the supplied evidence.
5. The application returns the answer and source information.

The generation prompt instructs the model to use the retrieved evidence and acknowledge when the available information is insufficient.

## 3. Main Project Files

| File | Responsibility |
|---|---|
| `main.py` | Runs PDF extraction, chunking, embedding generation, and database storage |
| `working_functions.py` | Contains helper functions for extracting text, creating chunks, and processing embeddings |
| `vector_store.py` | Handles Qdrant storage and retrieval |
| `rag_pipeline.py` | Connects question embedding, retrieval, and answer generation |
| `app.py` | Defines the FastAPI application and API endpoints |
| `config.py` | Loads the Gemini API key from the environment |
| `check_database.py` | Helps inspect the stored database |
| `search_test.py` | Tests retrieval from the vector database |
| `answer_test.py` | Tests the question-answering pipeline |
| `list_models.py` | Checks available Gemini models |

## 4. Processing and Checkpointing

Embedding generation requires API requests, so repeating the entire process after an interruption can waste time and API usage.

To reduce this problem, `main.py` uses an `embedding_checkpoint.json` file.

After successful embedding batches, the checkpoint saves the completed information.

When the script runs again, it:

1. Checks whether a checkpoint file exists.
2. Loads the saved chunks.
3. Compares the number of chunks and their text, source and page information with the current input.
4. Restores previously generated embeddings when the checkpoint matches.
5. Processes only chunks that do not already have embeddings.

If the checkpoint does not match the current paper collection, the script starts the embedding process again rather than blindly reusing old embeddings.

This is a basic recovery mechanism, not a complete database-versioning system.

## 5. Models and Configuration

| Component | Model or setting |
|---|---|
| Text embeddings | `gemini-embedding-001` |
| Embedding dimensions | 3,072 |
| Answer generation | `gemini-3.5-flash-lite` |
| Chunk size | 1,500 characters |
| Chunk overlap | 200 characters |
| Embedding batch size | 2 |
| Retrieved chunks | Top 3 |

The **embedding model** and **answer-generation model** have different responsibilities.

- The embedding model converts text into vectors for similarity search.
- The generation model uses the retrieved text to formulate an answer.

The generation model was changed during development after performance testing. The reasons and observations are documented below.

## 6. Testing and Evaluation

Testing was carried out manually using questions about the two included papers and questions that required information beyond the available documents.

The purpose was to check whether the application could retrieve relevant evidence, produce useful answers and recognise when the papers did not contain enough information.

### Questions about available evidence

The following nine questions tested information present in the research papers:

**Transformer paper**

1. What is the main idea behind the Transformer architecture?
2. Who are the authors of the paper *Attention Is All You Need*?
3. What problem does the Transformer architecture aim to solve?
4. Why is self-attention important in the Transformer?
5. How does multi-head attention help the Transformer capture different relationships between tokens?
6. Why does the Transformer use positional encoding, and what information does it provide?

**BERT paper**

7. What is the difference between BERT's pre-training and fine-tuning?
8. How does BERT use masked language modelling?
9. What are the two pre-training tasks used to train BERT?

### Questions testing insufficient evidence and broader claims

Five additional questions tested whether the application could avoid making claims beyond the available evidence:

1. How does the Transformer architecture perform image classification?
2. What are the main limitations of BERT when processing extremely long documents?
3. How much computational power is required to train GPT-4?
4. Can BERT outperform every other language model on every NLP task?
5. What is the current market share of ChatGPT among generative AI applications?

### Observations

The manual tests showed that:

- The application could answer questions about key concepts described in the papers.
- It could retrieve information from both included papers.
- Generated answers included paper and page references.
- It could qualify broad claims when the papers did not establish a universal conclusion.
- It could acknowledge insufficient evidence for questions outside the document collection.

One limitation became apparent: even when an answer correctly acknowledges insufficient evidence, Qdrant may still return the three closest available chunks. These chunks are not necessarily relevant enough to support an answer.

The current implementation does not use any minimum similarity-score threshold to filter weak retrieval matches.

### Evaluation scope

This was a small manual, qualitative evaluation. The answers and retrieved passages were inspected rather than measured against a formally annotated benchmark.

The project does not currently include formal retrieval metrics, a ground-truth answer dataset, or automated faithfulness evaluation.

## 7. Performance Testing and Model Selection

During development, the application initially used `gemini-3.8-flash` for answer generation.

Although question embedding was relatively quick, answer generation was unexpectedly slow.

The **generation request** took considerably longer than the question-embedding request:

| Test | Generation time |
|---|---:|
| Question 1 | 203.46 seconds |
| Question 2 | 103.44 seconds |
| Question 3 | 68.29 seconds |

**Question embedding** during these tests took approximately 0.67–0.78 seconds.

To investigate the delay, the generation model was changed to `gemini-3.5-flash-lite` (found from the list of available models after running `list_models.py`). The embedding model, vector database, retrieval settings, and prompts were kept unchanged.

After rebuilding the Docker image and recreating the container, the same three questions were tested again.

| Test | New generation time |
|---|---:|
| Question 1 | 2.28 seconds |
| Question 2 | 1.49 seconds |
| Question 3 | 2.07 seconds |

Question embedding took approximately 0.61–0.80 seconds.

This was a substantial improvement in the measured generation time, so `gemini-3.5-flash-lite` was retained.

Additional tests also showed that generation time could vary depending on the request. The measurements are observations from a small manual test set, not guaranteed response times.

**Important:** These timings measure the embedding and answer-generation API calls. They are not complete end-to-end response times, and Qdrant retrieval time was not separately benchmarked.

## 8. Debugging and Engineering Decisions

Several issues came up during implementation. Working through them helped me understand how the different parts of the application interact.

### A. Changing the generation model

The initial generation model was too slow for practical testing.

I switched to `gemini-3.5-flash-lite` after comparing the observed generation times. Since the model configuration is part of the application code, the Docker image had to be rebuilt before the updated version could run inside the container.

### B. Handling temporary API failures

Gemini API requests can occasionally fail because of temporary service or availability issues.

Retry settings were added to the Gemini clients in `rag_pipeline.py` to handle temporary API failures:

- Maximum attempts: 2 (one initial attempt and one retry)
- Initial retry delay: 1 second
- Maximum retry delay: 2 seconds

This allows a failed request to be attempted once more after a short delay. It helps with temporary failures but does not guarantee that every request will succeed.

The embedding workflow also includes retry and checkpoint handling through its processing function.

### C. Handling server errors in FastAPI

The application imports the Gemini API error class from `google.genai.errors`.

When a Gemini server error occurs, the API catches it and returns HTTP 503 (Service Unavailable) with a short message asking the user to try again later.

This is more useful than allowing the application to return an unhandled server error.

### D. Rebuilding Docker after code changes

Changes made to Python files on the computer are not automatically reflected in an already-built Docker image.

After modifying the application code, I rebuilt the image and recreated the container to test the updated version.

### E. Checking Docker logs

The following command was useful for inspecting the output of the running container:

```bash
docker logs research-agent
```

It helped inspect API errors and the measured embedding and generation times printed by the application.

## 9. Intended Use and Limitations

### Intended use

The application is intended as a small research-assistance and learning project.

Potential uses include:

- Exploring the contents of a selected collection of research papers.
- Finding passages related to a research question.
- Understanding technical concepts discussed in the included papers.
- Locating source pages for further reading.
- Experimenting with document retrieval and evidence-grounded question answering.

### Data used

The initial collection contains two publicly available research papers:

- *Attention Is All You Need*
- *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*

The processing workflow produced 92 text chunks:

| Paper | Number of chunks |
|---|---:|
| Attention Is All You Need | 38 |
| BERT | 54 |
| **Total** | **92** |

Text was extracted using PyMuPDF, divided into chunks and embedded using `gemini-embedding-001`.

The application uses the text available in these PDFs. It does not independently verify the accuracy of the original papers.

### Limitations

The current application has several limitations:

- **Small document collection:** It has only been tested with two research papers.
- **Limited retrieval:** It retrieves only the three closest chunks for each question.
- **No minimum relevance threshold:** Weakly related passages can still be returned.
- **Possible answer errors:** Gemini may misinterpret or overstate information in retrieved passages.
- **Citation limitations:** Source references identify retrieved paper pages, but do not guarantee that every generated statement is fully supported by the cited passage.
- **PDF extraction limitations:** Tables, figures, equations and complex layouts may not be represented accurately by extracted text.
- **External API dependency:** Embedding and generation depend on Gemini API availability, network access, and usage limits.
- **Performance variability:** Response times can change between requests.
- **Limited evaluation:** The project has not undergone a formal benchmark or independent review.

### Potential bias and edge cases

The application can inherit limitations from both its source documents and the Gemini models.

For example:

- **Limited viewpoints**: The application can only use the papers provided. It may miss other researchers' opinions or findings.
- **Missing information**: If the retrieved passages don't contain all the relevant details, the answer might be incomplete.
- **Overgeneralisation**: The application might make broad claims based on evidence that only supports a limited conclusion.
- **Outdated information**: It cannot automatically answer questions about recent developments using only older papers.
- **Irrelevant retrieval**: It might retrieve passages with similar keywords but a different meaning or context.

The application should therefore be treated as a research exploration tool, not a replacement for reading and evaluating original research.

### Out of scope

The application is not intended for:

- High-stakes decisions requiring verified evidence.
- Producing comprehensive or systematic literature reviews.
- Replacing expert judgement or academic source evaluation.
- Guaranteeing that every generated statement is factually correct.

## 10. Deployment Notes

The application supports local execution and Docker-based execution.

The Docker build includes the local Qdrant database so that the demonstration container can access the already-processed research papers.

For a new collection of papers, the database must be prepared for that collection before running the application. Avoid mixing old and new records unintentionally.

## 11. Possible Future Improvements

The project is intentionally kept small, but several improvements could be explored later:

- Add a minimum similarity threshold to filter weak retrieval results.
- Test retrieval quality with a larger and more varied collection of papers.
- Improve handling of tables, figures and other PDF elements.
- Evaluate answer quality using a more structured test dataset.
- Improve citation verification.
- Add a simple user interface for submitting questions.

These are possible next steps, not features currently implemented.

## 12. Learning and Acknowledgements

This project was developed as a practical learning exercise while exploring AI engineering concepts, particularly embeddings, vector databases, RAG pipelines, API development, testing, and Docker.

ChatGPT was used as a learning and development assistant to help explain concepts, discuss implementation approaches, troubleshoot errors and refine documentation. The project was built and tested through hands-on work and iterative problem-solving.