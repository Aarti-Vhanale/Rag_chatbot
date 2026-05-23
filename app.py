"""
app.py - FastAPI Web Server
Serves the RAG chatbot API + serves the chat UI
"""

import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from generator import generate_answer
from retriever import retrieve, get_collection

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="1.0")

# Serve static UI files
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


class QueryRequest(BaseModel):
    query: str
    use_reranker: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query: str
    latency_ms: float


@app.get("/")
async def root():
    """Serve the chat UI."""
    return FileResponse("ui/index.html")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main RAG endpoint.
    POST /query
    Body: {"query": "your question here"}
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start = time.time()
    
    try:
        result = generate_answer(request.query, use_reranker=request.use_reranker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    latency_ms = (time.time() - start) * 1000
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query=result["query"],
        latency_ms=round(latency_ms, 2)
    )


@app.get("/retrieve")
async def retrieve_endpoint(query: str, top_k: int = 5):
    """
    Debug endpoint: see raw retrieved chunks without generation.
    GET /retrieve?query=your+question&top_k=5
    """
    chunks = retrieve(query, top_k=top_k)
    return {"query": query, "chunks": chunks}


@app.get("/stats")
async def stats_endpoint():
    """Get DB statistics."""
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "vector_db": "ChromaDB (HNSW)",
        "llm": "Groq llama3-8b (FREE)"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)