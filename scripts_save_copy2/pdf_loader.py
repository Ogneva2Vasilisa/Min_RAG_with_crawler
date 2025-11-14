# -*- coding: utf-8 -*-
from pathlib import Path
import requests
import pdfplumber
from typing import List
from scripts.model_init import add_chunks_to_faiss

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text_from_pdf_file(path: Path) -> str:
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
    return "\n\n".join(texts)

def extract_text_from_pdf_url(url: str) -> str:
    import io
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            texts = []
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
            return "\n\n".join(texts)
    except Exception as e:
        print(f"[ERROR] Failed to fetch PDF from {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


    
def add_pdfs_to_faiss_main(pdf_dir: str, output_dir: str, embedder):
    """
    Главная функция для добавления PDF из локальной папки в FAISS.
    """

    pdf_dir = Path(pdf_dir)
    items = {}
    for f in pdf_dir.rglob("*.pdf"):
        text = extract_text_from_pdf_file(f)
        if text.strip():
            items[str(f.resolve())] = {"text": text, "title": f.stem}

    if not items:
        print("[INFO] PDF файлов для добавления не найдено.")
        return

    add_chunks_to_faiss(items, output_dir, embedder)