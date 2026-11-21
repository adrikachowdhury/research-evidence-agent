# Research Evidence Agent

A beginner-built, multi-paper research assistant using Retrieval-Augmented Generation (RAG), Gemini, Qdrant, and FastAPI.

Research papers can be lengthy, and finding specific information across multiple papers can take time. This project explores how a RAG-based application can retrieve relevant passages from research papers and use them to generate answers grounded in the available evidence.

The application currently works with **two research papers**, `Attention Is All You Need` and `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`. Users can submit questions through an API and receive answers with references to the source papers and page numbers.

## Features

- Extracts text and page information from PDF research papers.
- Splits extracted text into smaller, overlapping chunks.
- Generates text embeddings using **Google's Gemini API**.
- Stores and retrieves relevant chunks using a local **Qdrant** database.
- Generates answers based on retrieved research evidence.
- Includes source paper names and page references in responses.
- Handles questions when the available documents do not contain sufficient evidence.
- Provides a REST API using **FastAPI**.
- Supports containerised execution using **Docker**.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF processing | PyMuPDF |
| Embedding model | `gemini-embedding-001` |
| Answer-generation model | `gemini-3.5-flash-lite` |
| Vector database | Qdrant |
| API framework | FastAPI |
| Data validation | Pydantic |
| Containerisation | Docker |

## How It Works

The application follows a basic RAG workflow:

1. **Extract:** Reads PDF files and extracts their text and page information.
2. **Chunk:** Splits the text into smaller sections with some overlap.
3. **Embed:** Converts the chunks into numerical vector representations.
4. **Store:** Saves the vectors and their associated text and source information in Qdrant.
5. **Retrieve:** Finds the three most relevant chunks for a user's question.
6. **Augment:** Combines the user's question with the retrieved chunks to provide relevant research evidence as context.
7. **Generate:** Gemini uses the question and retrieved context to generate an answer.

The answer-generation process is designed to use the **retrieved passages** as its evidence rather than answer solely from the language model's general knowledge.

### Using Your Own Research Papers

You can replace or add PDF files inside `data/papers/`. The processing script automatically detects PDF files in this folder.

When changing the collection, make sure the existing Qdrant database is appropriately cleared or rebuilt first. Otherwise, old records may remain in the database. So. make sure to delete the existing `qdrant_db/` folder and `embedding_checkpoint.json` before building the database if you're replacing my paper collection.

## Project Structure
```
research-evidence-agent/
│
├── data/
│   └── papers/                    # Research PDFs used by the application
│       ├── attention_is_all_you_need.pdf
│       └── bert.pdf
│
├── main.py                        # Extracts, chunks, and embeds PDF text
├── working_functions.py           # Helper functions for document processing
├── vector_store.py                # Stores and retrieves chunks using Qdrant
├── rag_pipeline.py                # Handles retrieval and answer generation
├── app.py                         # FastAPI application and API endpoints
├── config.py                      # Loads the Gemini API key
│
├── scripts/
│   ├── check_database.py          # Checks the contents of the vector database
│   ├── search_test.py             # Tests document retrieval
│   ├── answer_test.py             # Tests generated answers
│   └── list_models.py             # Lists available Gemini models
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Instructions for building the Docker image
├── .dockerignore                  # Files excluded from the Docker build
├── .gitignore                     # Files excluded from Git
│
├── README.md                      # Project overview and setup instructions
│
└── docs/
    ├── PROJECT_GUIDE.md           # Implementation, testing, and technical details
    └── GLOSSARY.md                # Beginner-friendly technical definitions
```
## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/adrikachowdhury/research-evidence-agent.git
cd research-evidence-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The key is loaded through `config.py`. Keep your `.env` file private and do not commit it to GitHub.

### 5. Process the research papers

Run:

```bash
python main.py
```

This processes the PDFs inside `data/papers/` and performs the following steps:

**PDF extraction → Text chunking → Embedding generation → Qdrant storage**

The current configuration uses:

- Chunk size: 1,500 characters
- Chunk overlap: 200 characters
- Embedding batch size: 2

The script also saves completed embeddings in a **checkpoint** file to help avoid repeating successful embedding requests if processing is interrupted.

### 6. Start the API

```bash
uvicorn app:app --reload
```

The application will be available at:

`http://127.0.0.1:8000`

To explore and test the API, open:

`http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Checks whether the API is running |
| POST | `/ask` | Submits a research question |

Example request:

```json
{
    "question": "What is the main idea behind the Transformer architecture?"
}
```

The response contains a generated answer and source information, including the paper name and page number. Example answer:

```json
{
  "answer": "Based on the provided evidence, the main idea behind the Transformer architecture is to eschew recurrence and instead rely entirely on an attention mechanism (specifically self-attention) to draw global dependencies between input and output, without using sequence-aligned RNNs or convolution (*papers\\attention_is_all_you_need.pdf*, p. 2, 10).",
  "sources": [
    {
      "source": "papers\\attention_is_all_you_need.pdf",
      "page": 2
    },
    {
      "source": "papers\\attention_is_all_you_need.pdf",
      "page": 10
    },
    {
      "source": "papers\\attention_is_all_you_need.pdf",
      "page": 2
    }
  ]
}
```

## Running with Docker

First, generate the local Qdrant database by completing the processing step above.

Build the Docker image:
```bash
docker build -t research-evidence-agent .
```

Run the container:
```bash
docker run -d -p 8000:8000 --env-file .env --name research-agent research-evidence-agent
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

Useful commands:
```bash
# View container logs
docker logs --tail 50 research-agent

# Stop the container
docker stop research-agent

# Remove the container
docker rm research-agent
```

## Documentation

- [Project Guide](docs/PROJECT_GUIDE.md): Implementation details, architecture, testing, debugging decisions, and limitations.
- [Glossary](docs/GLOSSARY.md): Beginner-friendly explanations of the technical terms used in this project.

## Acknowledgements

I used **ChatGPT** as a learning and development assistant throughout this project. It helped me understand unfamiliar concepts, work through implementation and debugging issues, and improve the project documentation.

The project was built and tested as part of my own hands-on learning process.