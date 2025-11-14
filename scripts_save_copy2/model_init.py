#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инициализация модели и эмбеддингов для проекта RAG
"""
import os
import requests
from langchain_openai import OpenAI


USER_AGENT = os.environ.get("USER_AGENT", "rag-crawler/1.0")

# LM Studio настройки
LM_API_URL = "http://localhost:1234/v1"
LM_API_KEY = "lm-studio"
EMBEDDING_MODEL = "text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf"
LLM_MODEL_NAME = "Qwen2.5-3B-Instruct"
EMBEDDING_MODEL_NAME = "text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf"
FAISS_INDEX_NAME = "faiss_index"  #!!!
METADATA_NAME = "metadata.json"  #!!!



class LMStudioEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, api_url=LM_API_URL, api_key=LM_API_KEY):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(f"{self.api_url}/embeddings", json=payload, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        return self.embed_query(text)


def get_embedder():
    return LMStudioEmbeddings(EMBEDDING_MODEL, LM_API_URL, LM_API_KEY)


def get_llm():
    return OpenAI(
        openai_api_base=LM_API_URL,
        openai_api_key=LM_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.3
    )
    
def get_faiss_path(kb_path):
    return os.path.join(kb_path, FAISS_INDEX_NAME)

def get_metadata_path(kb_path):
    return os.path.join(kb_path, METADATA_NAME)

#--------------------
from pathlib import Path
import os
import hashlib
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def add_chunks_to_faiss(items: dict, output_dir: str, embedder, min_text_len: int = 30):
    """
    Унифицированная функция для добавления чанков в FAISS.
    
    items: dict
        {source_identifier: {"text": ..., "title": ...}}
        source_identifier может быть URL или путь к PDF.
    
    output_dir: str
        Папка для FAISS и метаданных.
    
    embedder: LMStudioEmbeddings
        Эмбеддер.
    
    min_text_len: int
        Минимальная длина текста для добавления.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    faiss_dir = get_faiss_path(output_dir)
    existing_hashes = set()
    db = None

    # Загрузка существующей базы
    if os.path.exists(faiss_dir):
        db = FAISS.load_local(faiss_dir, embedder)
        for m in getattr(db, "metadatas", []) or []:
            txt = m.get("text", "")
            if txt:
                existing_hashes.add(hashlib.sha1(txt.encode("utf-8")).hexdigest())
        print(f"[INFO] Загружена существующая FAISS база с {len(existing_hashes)} чанками.")

    all_chunks = []
    all_metadatas = []

    for source, data in items.items():
        text = data.get("text", "")
        title = data.get("title", source)
        if not text or len(text.strip()) < min_text_len:
            continue

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip() or len(chunk.strip()) < min_text_len:
                continue
            uid = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
            if uid in existing_hashes:
                continue
            existing_hashes.add(uid)
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": source,
                "title": f"{title} (chunk {i})",
                "chunk_id": i,
                "text": chunk
            })

    if not all_chunks:
        print("[INFO] Нет новых чанков для добавления.")
        return db

    print(f"[INFO] Добавление {len(all_chunks)} чанков в FAISS...")
    if db is None:
        db = FAISS.from_texts([], embedder)
    for chunk, meta in tqdm(zip(all_chunks, all_metadatas), total=len(all_chunks), desc="Adding chunks to FAISS"):
        db.add_texts([chunk], metadatas=[meta])

    os.makedirs(faiss_dir, exist_ok=True)
    db.save_local(faiss_dir)
    print(f"[OK] FAISS сохранён в {faiss_dir}")
    return db
