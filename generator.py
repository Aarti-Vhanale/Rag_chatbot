"""
generator.py - RAG Answer Generation
Uses Groq (FREE tier - llama3) to generate answers from retrieved context
"""

import os
from groq import Groq
from dotenv import load_dotenv
from retriever import retrieve, format_context
from reranker import rerank

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K", 5))

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions ONLY based on the provided document excerpts.

Rules:
1. Answer only from the provided context
2. Always cite your sources as: [Source: filename, Page X]
3. If the answer is not in the context, say "I couldn't find this in the provided documents"
4. Be concise and accurate
5. Never make up information"""


def build_prompt(query: str, context: str) -> str:
    return f"""Based on the following document excerpts, answer the question.

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

ANSWER (cite sources inline like [Source: filename.pdf, Page X]):"""


def generate_answer(query: str, use_reranker: bool = True) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Optionally rerank
    3. Generate answer with citations
    
    Returns:
    {
        "answer": str,          ← Generated answer with citations
        "sources": list[dict],  ← Source chunks used
        "query": str            ← Original query
    }
    """
    # Step 1: Retrieve
    raw_chunks = retrieve(query, top_k=TOP_K)
    
    # Step 2: Rerank (optional but improves quality)
    if use_reranker and raw_chunks:
        try:
            chunks = rerank(query, raw_chunks, top_n=3)
        except Exception:
            chunks = raw_chunks[:3]  # Fallback if reranker unavailable
    else:
        chunks = raw_chunks[:3]
    
    # Step 3: Build context
    context = format_context(chunks)
    prompt = build_prompt(query, context)
    
    # Step 4: Generate answer via Groq (FREE llama3)
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # FREE on Groq
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.1  # Low temp for factual accuracy
    )
    
    answer = response.choices[0].message.content
    
    # Build unique source list
    seen = set()
    sources = []
    for chunk in chunks:
        key = f"{chunk['filename']}_p{chunk['page_number']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "score": chunk.get("rerank_score", chunk.get("score", 0)),
                "preview": chunk["text"][:200] + "..."
            })
    
    return {
        "answer": answer,
        "sources": sources,
        "query": query,
        "chunks_used": len(chunks)
    }


if __name__ == "__main__":
    result = generate_answer("Explain the main topic of the document")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['filename']}, Page {s['page_number']}")