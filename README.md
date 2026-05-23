# RAG Chatbot — Complete Setup Guide

## Tech Stack (All FREE/Open-source)
| Component | Tool | Cost |
|-----------|------|------|
| PDF Extraction | PyMuPDF | Free |
| OCR | Tesseract | Free |
| Embeddings | all-MiniLM-L6-v2 | Free (local) |
| Vector DB | ChromaDB (HNSW) | Free (local) |
| Reranker | cross-encoder MiniLM | Free (local) |
| LLM | Groq llama3-8b | Free (API) |

## Setup Steps

### Step 1: Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 2: Create Python Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# OR
venv\Scripts\activate         # Windows
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get FREE Groq API Key
1. Go to https://console.groq.com
2. Sign up (free)
3. Create API key
4. Add to .env file: `GROQ_API_KEY=your_key`

### Step 5: Add PDFs
```bash
mkdir -p data/pdfs
# Copy your 10+ PDFs (200+ pages each) into data/pdfs/
```

### Step 6: Run Ingestion (One time only)
```bash
python embedder.py
# This will:
# - Extract text from all PDFs
# - Run OCR on scanned pages
# - Chunk text
# - Generate embeddings
# - Store in ChromaDB
# Takes 10-30 mins for 10 large PDFs
```

### Step 7: Start the Server
```bash
python app.py
# Server runs at http://localhost:8000
```

### Step 8: Open Chat UI