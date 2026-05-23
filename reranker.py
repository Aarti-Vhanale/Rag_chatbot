"""
reranker.py - Cross-encoder reranking for better precision
Reranks top-K retrieved chunks using a cross-encoder model
"""

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Lightweight cross-encoder (FREE, 22MB)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
    """
    Rerank retrieved chunks using cross-encoder for better relevance.
    Returns top_n chunks sorted by reranker score.
    """
    if not chunks:
        return chunks
    
    reranker = get_reranker()
    
    # Prepare query-passage pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Get reranker scores
    scores = reranker.predict(pairs)
    
    # Add reranker scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]