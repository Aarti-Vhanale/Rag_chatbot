"""
ingest.py - PDF Ingestion Pipeline
Extracts text from PDFs (native + OCR), chunks it, and stores in ChromaDB
"""

import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import io

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))


def extract_text_from_page(page) -> str:
    """
    Extract text from a PDF page.
    Uses native text first; falls back to OCR if text is empty/scanned.
    """
    # Try native text extraction
    text = page.get_text("text")
    
    if len(text.strip()) < 50:  # Likely scanned page
        # Convert page to image and OCR it
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang='eng')
    
    return text


def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Remove excessive whitespace
    - Remove headers/footers patterns
    - Normalize unicode
    """
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove excessive spaces
    text = re.sub(r' {3,}', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    chunk_size: approximate word count per chunk
    overlap: word count overlap between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Move forward with overlap
    
    return chunks


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Process a single PDF:
    Returns list of chunk dicts with text + metadata
    """
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).name
    all_chunks = []
    
    print(f"\nProcessing: {filename} ({len(doc)} pages)")
    
    for page_num in tqdm(range(len(doc)), desc=f"  Pages"):
        page = doc[page_num]
        raw_text = extract_text_from_page(page)
        clean = clean_text(raw_text)
        
        if len(clean.strip()) < 20:  # Skip near-empty pages
            continue
        
        chunks = chunk_text(clean)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "pdf_id": filename.replace(".pdf", ""),
                "filename": filename,
                "page_number": page_num + 1,  # 1-indexed
                "chunk_index": i,
                "chunk_id": f"{filename}__pg{page_num+1}__ch{i}"
            })
    
    doc.close()
    print(f"  → {len(all_chunks)} chunks extracted from {filename}")
    return all_chunks


def ingest_all_pdfs() -> list[dict]:
    """
    Process all PDFs in PDF_DIR.
    Returns all chunks across all PDFs.
    """
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDFs found in {PDF_DIR}. Add PDFs and retry.")
    
    print(f"Found {len(pdf_files)} PDFs to ingest:")
    for f in pdf_files:
        print(f"  - {f.name}")
    
    all_chunks = []
    for pdf_path in pdf_files:
        chunks = process_pdf(str(pdf_path))
        all_chunks.extend(chunks)
    
    print(f"\n✅ Total chunks extracted: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_all_pdfs()
    print(f"\nFirst chunk preview:")
    print(chunks[0]["text"][:300])
    print(f"Metadata: {chunks[0]['filename']} | Page {chunks[0]['page_number']}")