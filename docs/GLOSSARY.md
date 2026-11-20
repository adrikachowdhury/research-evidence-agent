# Glossary

Beginner-friendly explanations of the technical terms used in the Research Evidence Agent.

| Term | Meaning |
|---|---|
| **API (Application Programming Interface)** | A way for two software applications to communicate. In this project, FastAPI provides an interface through which users can submit questions. |
| **Batch Processing** | Processing multiple items together instead of handling each item individually. The embedding workflow processes chunks in batches of two. |
| **Chunk** | A smaller section of text created by splitting a larger document. The application divides extracted research paper text into chunks before generating embeddings. |
| **Chunk Overlap** | Text shared between neighbouring chunks. It helps preserve context when a sentence or idea crosses the boundary between two chunks. |
| **Docker** | A tool that packages an application and its dependencies into a container so it can run in a more consistent environment. |
| **Embedding** | A numerical representation of text. Text with similar meanings can have similar vector representations, making it useful for similarity-based retrieval. |
| **FastAPI** | A Python framework used to build APIs. It allows the Research Evidence Agent to receive questions and return answers through HTTP endpoints. |
| **Gemini** | Google's family of AI models. This project uses one Gemini model for embeddings and another for answer generation. |
| **HTTP Status Code** | A number returned by a web server to describe the result of a request. For example, 200 generally means success, while 503 means the service is temporarily unavailable. |
| **JSON** | A lightweight format for storing and exchanging structured data. The API accepts questions in JSON format and returns structured responses. |
| **Language Model** | An AI model trained to process and generate human language. The answer-generation component uses a Gemini language model. |
| **Latency** | The time taken for an operation or request to complete. In this project, embedding and answer-generation times were measured separately. |
| **LLM (Large Language Model)** | A language model trained on large amounts of text to perform tasks such as answering questions, summarising and generating text. |
| **Metadata** | Additional information describing a piece of data. Here, metadata includes the source PDF name and page number associated with a text chunk. |
| **Pydantic** | A Python library used for data validation. FastAPI uses it to check the structure and types of incoming API data. |
| **PyMuPDF** | A Python library used to read and extract information from PDF files. |
| **Qdrant** | A vector database that stores and searches vector representations. This project uses it to find research paper passages relevant to a question. |
| **RAG (Retrieval-Augmented Generation)** | An approach that retrieves relevant information from external documents and provides it to a language model as context for generating an answer. |
| **REST API** | A common way for applications to communicate over HTTP using methods such as GET and POST. |
| **Retrieval** | The process of finding and returning relevant information from a collection of stored documents. |
| **Similarity Search** | A search method that finds items whose vector representations are close to a query vector. It helps retrieve passages related to a user's question. |
| **Source Attribution** | Identifying where information came from. The application returns the source paper and page number associated with retrieved passages. |
| **Vector** | A list of numbers representing information. An embedding is a vector that represents text numerically. |
| **Vector Database** | A database designed to store and search vector representations efficiently. Qdrant serves this purpose in the project. |

## Concepts in This Project

### 1. Cross-document retrieval

Finding relevant information across more than one document.

For example, a question about the differences between Transformer and BERT architectures could require retrieving passages from both papers.

### 2. Evidence-grounded question answering

Generating an answer using information retrieved from the available documents.

The goal is to make the answer rely on the supplied research evidence rather than simply asking the language model to answer from its general knowledge.

### 3. Insufficient-evidence recognition

Recognising when the available documents do not contain enough information to answer a question reliably.

For example, the two included papers cannot establish the current market share of ChatGPT. The application should acknowledge this limitation rather than invent a figure.

### 4. Checkpointing

Saving completed work so that it does not need to be repeated unnecessarily.

In this project, generated embeddings are saved in a checkpoint file. If processing is interrupted, the script can reuse previously completed embeddings when the saved data matches the current chunks.

### 5. Model latency comparison

Measuring and comparing how long different models take to perform a task.

The project compared two Gemini answer-generation models and retained the faster model based on the observed manual test results.

### 6. Model limitations and bias

A model's output can be affected by the information it receives, the data it was trained on and the way a question is phrased.

In this application, limitations can also come from the research papers themselves, incomplete retrieval and the language model's interpretation of the retrieved passages.