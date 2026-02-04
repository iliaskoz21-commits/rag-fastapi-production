import os
import pickle
from utils import load_documents, split_documents
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# INGEST DOCS FUNCTION
# -------------------------
def ingest_docs(files, docs_folder, chunk_file, index_file, chunk_size=300, chunk_overlap=50):
    # Φτιάξε docs folder αν δεν υπάρχει
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)

    # Αποθήκευση των uploaded αρχείων στο docs folder
    for f in files:
        file_path = os.path.join(docs_folder, f.filename)
        with open(file_path, "wb") as out_file:
            out_file.write(f.file.read())

    # Φόρτωσε και σπάσε όλα τα docs
    documents = load_documents(docs_folder)
    chunk_texts = split_documents(documents, chunk_size, chunk_overlap)

    # Δημιούργησε embeddings & FAISS index
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Αποθήκευση cache σε αρχεία
    with open(chunk_file, "wb") as f:
        pickle.dump(chunk_texts, f)
    with open(index_file, "wb") as f:
        pickle.dump(index, f)

    return chunk_texts, index, embedding_model
