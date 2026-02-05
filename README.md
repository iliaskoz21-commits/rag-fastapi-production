# Hybrid RAG FastAPI Agent

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.101-green)
![Build](https://img.shields.io/github/workflow/status/iliaskoz21-commits/rag-fastapi-production/CI)

## Introduction
This project implements a **Hybrid RAG (Retrieval-Augmented Generation) Agent**:

- **Vector search** using FAISS + SentenceTransformers embeddings  
- **Keyword search** over documents  
- **LLM**: Ollama (offline Mistral model)  
- **FastAPI web app** with endpoints and HTML chat interface  
- **Document ingestion** on the fly with automatic embedding & index update  

---

## Demo

Open in browser:

[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Type a question in the input box and click **Send**.

API example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"Explain RAG\"}"
Example response:

{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique..."
}
Ingest new documents:

curl -X POST "http://127.0.0.1:8000/ingest" -F "files=@example.pdf"
Installation & Run
Locally (Python)
git clone https://github.com/iliaskoz21-commits/rag-fastapi-production.git
cd rag-fastapi-production

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload
Open browser: http://127.0.0.1:8000/

Docker
docker build -t rag-fastapi-agent .
docker run -p 8000:8000 rag-fastapi-agent
API Endpoints
Method	Path	Request	Response
GET	/	—	HTML chat interface
POST	/query	{ "question": "..." }	{ "answer": "..." }
POST	/ingest	List of files as multipart/form-data	{ "message": "Documents ingested and index updated." }
Deployment / Cloud (Optional)
You can deploy this app on:

Fly.io

Render

Railway

AWS ECS / Fargate

💡 Tip: Make sure docs folder and FAISS index files are persisted.

Technologies
Python 3.11

FastAPI

Ollama (Mistral)

FAISS + SentenceTransformers

Uvicorn

LangChain document loaders

Pytest