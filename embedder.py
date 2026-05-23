"""
embedder.py - Embedding Generation & Vector DB Storage
Uses sentence-transformers (FREE, runs locally)
Stores in ChromaDB (FREE, open-source)
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "rag_documents"
BATCH_SIZE = 100  # Process embeddings in batches


def get_embedding_model():
    """Load the sentence-transformer embedding model (downloads once, cached)."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_chroma_client():
    """Initialize persistent ChromaDB client."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection(client):
    """Get existing collection or create new one with HNSW index."""
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✅ Loaded existing collection: {COLLECTION_NAME} ({collection.count()} docs)")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",     # Cosine similarity
                "hnsw:construction_ef": 200, # Build quality (higher = better, slower)
                "hnsw:M": 16                 # Connections per node
            }
        )
        print(f"✅ Created new collection: {COLLECTION_NAME}")
    return collection


def embed_and_store(chunks: list[dict], model, collection):
    """
    Generate embeddings for all chunks and store in ChromaDB.
    Processes in batches to avoid memory issues.
    """
    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    
    # Check which chunks are already in DB (avoid re-embedding)
    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    
    if not new_chunks:
        print("✅ All chunks already embedded. Skipping.")
        return
    
    print(f"Embedding {len(new_chunks)} new chunks...")
    
    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = new_chunks[i:i + BATCH_SIZE]
        
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "pdf_id": c["pdf_id"],
            "filename": c["filename"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"]
        } for c in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        
        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    print(f"✅ Total stored in ChromaDB: {collection.count()} chunks")


def run_ingestion_pipeline():
    """Full pipeline: PDF → chunks → embeddings → ChromaDB."""
    from ingest import ingest_all_pdfs
    
    # Step 1: Extract chunks from PDFs
    chunks = ingest_all_pdfs()
    
    # Step 2: Load embedding model
    model = get_embedding_model()
    
    # Step 3: Initialize ChromaDB
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    # Step 4: Embed and store
    embed_and_store(chunks, model, collection)
    
    print("\n🎉 Ingestion pipeline complete!")
    print(f"   Vector DB location: {CHROMA_DB_PATH}")
    print(f"   Total chunks in DB: {collection.count()}")
    return collection


if __name__ == "__main__":
    run_ingestion_pipeline()