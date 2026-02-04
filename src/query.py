from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# HYBRID SEARCH FUNCTION
# -------------------------
def hybrid_search(query, chunk_texts, index, embedding_model, vector_k=5, keyword_k=5):
    query_clean = query.lower()

    # 🔍 Keyword search
    keyword_hits = [
        text for text in chunk_texts
        if any(word in text for word in query_clean.split())
    ][:keyword_k]

    # 🧠 Vector search
    q_emb = embedding_model.encode([query_clean], convert_to_numpy=True)
    _, idxs = index.search(q_emb, vector_k)
    vector_hits = [chunk_texts[i] for i in idxs[0]]

    # 🔗 Merge (unique)
    combined = []
    for t in keyword_hits + vector_hits:
        if t not in combined:
            combined.append(t)

    return combined
