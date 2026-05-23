"""
retriever.py - Vector Similarity Search
Retrieves top-K relevant chunks from ChromaDB for a user query
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "rag_documents"
TOP_K = int(os.getenv("TOP_K", 5))

# Singleton pattern: load model once
_model = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-K most relevant chunks for a query.
    
    Returns list of dicts:
    {
        "text": str,          ← chunk content
        "filename": str,      ← source PDF filename
        "page_number": int,   ← page in that PDF
        "score": float        ← cosine similarity (0-1, higher=better)
    }
    """
    model = get_model()
    collection = get_collection()
    
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search ChromaDB (returns top-K closest vectors)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    chunks = []
    for i in range(len(results["ids"][0])):
        # ChromaDB returns L2 distance; convert to similarity score
        distance = results["distances"][0][i]
        similarity = 1 - distance  # For cosine space in ChromaDB
        
        chunks.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "pdf_id": results["metadatas"][0][i]["pdf_id"],
            "chunk_id": results["ids"][0][i],
            "score": round(similarity, 4)
        })
    
    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    Includes source citations inline.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']}, Page {chunk['page_number']}]\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


if __name__ == "__main__":
    # Test retrieval
    query = "What is machine learning?"
    results = retrieve(query)
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['filename']} p.{r['page_number']}")
        print(f"  {r['text'][:150]}...")
        print()