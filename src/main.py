import os
import warnings
import pickle
import yaml
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import ollama

from sentence_transformers import SentenceTransformer
import faiss

from utils import load_documents, split_documents
from ingest import ingest_docs
from query import hybrid_search

# =========================
# LOAD CONFIG
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DOCS_FOLDER = config["docs_folder"]
CHUNK_DATA_FILE = config["chunk_data_file"]
VECTOR_INDEX_FILE = config["vector_index_file"]
VECTOR_K = config.get("vector_k", 5)
KEYWORD_K = config.get("keyword_k", 5)
EMBEDDING_MODEL = config.get("embedding_model")
OLLAMA_MODEL = config.get("ollama_model", "mistral")
OLLAMA_TEMP = config.get("ollama_temperature", 0.2)

# =========================
# LOAD OR INITIALIZE INDEX
# =========================
if os.path.exists(CHUNK_DATA_FILE) and os.path.exists(VECTOR_INDEX_FILE):
    with open(CHUNK_DATA_FILE, "rb") as f:
        chunk_texts = pickle.load(f)
    with open(VECTOR_INDEX_FILE, "rb") as f:
        index = pickle.load(f)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
else:
    docs = load_documents(DOCS_FOLDER)
    chunk_texts = split_documents(docs)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    # Save cache
    with open(CHUNK_DATA_FILE, "wb") as f:
        pickle.dump(chunk_texts, f)
    with open(VECTOR_INDEX_FILE, "wb") as f:
        pickle.dump(index, f)

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Hybrid RAG Agent</title>
<style>
body { font-family: Arial; max-width: 700px; margin: auto; padding: 20px;}
#chat { border:1px solid #ccc; padding:10px; height:400px; overflow-y:scroll; margin-bottom: 10px;}
.user { color: blue; margin:5px 0; }
.bot { color: green; margin:5px 0; }
#q { width:80%; padding:10px; }
button { padding: 10px; }
</style>
</head>
<body>
<h2>Hybrid RAG Agent (Keyword + Vector)</h2>

<div id="chat"></div>
<input id="q" placeholder="Type your question...">
<button onclick="ask()">Send</button>

<script>
async function ask() {
    let q = document.getElementById("q").value;
    if(!q) return;

    let chat = document.getElementById("chat");
    chat.innerHTML += "<div class='user'>You: " + q + "</div>";
    document.getElementById("q").value = "";

    try {
        let r = await fetch("/query", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body: JSON.stringify({question:q})
        });
        let d = await r.json();
        chat.innerHTML += "<div class='bot'>Bot: " + d.answer + "</div>";
        chat.scrollTop = chat.scrollHeight;
    } catch(err) {
        chat.innerHTML += "<div class='bot'>Error: " + err + "</div>";
    }
}
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.post("/query")
async def query(req: Request):
    q = (await req.json()).get("question", "")
    contexts = hybrid_search(q, chunk_texts, index, embedding_model, VECTOR_K, KEYWORD_K)
    context = "\n\n".join(contexts)
    prompt = f"""
You are an expert AI assistant.
Answer ONLY in English.
Use ONLY the context.
If the answer is not present, say "I don't know".

CONTEXT:
{context}

QUESTION:
{q}
"""
    r = ollama.chat(
        model=OLLAMA_MODEL,
        options={"temperature": OLLAMA_TEMP},
        messages=[{"role":"user","content":prompt}]
    )
    return JSONResponse({"answer": r["message"]["content"]})

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    global chunk_texts, index, embedding_model
    chunk_texts, index, embedding_model = ingest_docs(
        files, DOCS_FOLDER, CHUNK_DATA_FILE, VECTOR_INDEX_FILE
    )
    return {"message": "Documents ingested and index updated."}

# =========================
# RUN
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
