# -*- coding: utf-8 -*-
from pathlib import Path
import requests
import pdfplumber
from typing import List

class LMStudioEmbeddings:
    """Простейшая обёртка для LM Studio эмбеддингов"""
    def __init__(self, model_name: str, api_url: str, api_key: str):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        import json
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(f"{self.api_url}/embeddings", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

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
