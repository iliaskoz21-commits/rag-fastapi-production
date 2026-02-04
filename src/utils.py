import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.lower()

# -------------------------
# LOAD DOCUMENTS
# -------------------------
def load_documents(docs_folder: str):
    documents = []
    for file in os.listdir(docs_folder):
        path = os.path.join(docs_folder, file)
        if file.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())
        elif file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    # Clean text
    for d in documents:
        d.page_content = clean_text(d.page_content)
    return documents

# -------------------------
# SPLIT DOCUMENTS
# -------------------------
def split_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    chunk_texts = [c.page_content for c in chunks]
    return chunk_texts
