#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG FAISS Builder с поддержкой:
- HTML страниц (BFS обход, max_pages)
- PDF файлов (локальные + ссылки на PDF)
- Обновление базы FAISS без дубликатов
- urls.txt с уникальными URL
- Фильтрация URL по seed-доменам, исключение mailto/tel
"""

import argparse
import os
import time
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---- Настройки ----
DEFAULT_OUT = "kb_output"
DEFAULT_FAISS_DIR = os.path.join(DEFAULT_OUT, "faiss_index")
DEFAULT_URLS_FILE = "urls.txt"
DEFAULT_SEEDS_FILE = "seed_urls.txt"

LM_API = "http://localhost:1234/v1"
LM_API_KEY = "lm-studio"
EMBEDDING_MODEL = "text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
USER_AGENT = os.environ.get("USER_AGENT", "rag-crawler/1.0")


# ----- Вспомогательные функции -----
def normalize_url(raw_url):
    parsed = urlparse(raw_url.strip())
    if parsed.scheme == "":
        parsed = parsed._replace(scheme="http")
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


# ----- LM Studio Embeddings -----
class LMStudioEmbeddings:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(f"{self.api_url}/embeddings", json=payload, headers=self.headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        return self.embed_query(text)


# ----- PDF обработка -----
import pdfplumber

def extract_text_from_pdf_file(path: Path):
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {path}: {e}")
    return "\n\n".join(texts)

def extract_text_from_pdf_url(url):
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        with open("tmp.pdf", "wb") as f:
            f.write(resp.content)
        text = extract_text_from_pdf_file(Path("tmp.pdf"))
        os.remove("tmp.pdf")
        return text
    except Exception as e:
        print(f"[ERROR] Failed to read PDF from URL {url}: {e}")
        return ""


# ----- Фильтрация URL по seed-доменам -----
def get_seed_domains(seeds):
    domains = set()
    for s in seeds:
        parsed = urlparse(normalize_url(s))
        domains.add(f"{parsed.scheme}://{parsed.netloc}")
    return domains

def is_allowed_url(url, seed_domains):
    if url.startswith("mailto:") or url.startswith("tel:"):
        return False
    parsed = urlparse(url)
    for sd in seed_domains:
        if url.startswith(sd):
            return True
    return False


# ----- BFS Crawl -----
def crawl(seeds, max_pages=None, delay=0.2):
    seed_domains = get_seed_domains(seeds)
    queue = deque(normalize_url(s) for s in seeds)
    seen = set()
    ordered_urls = []
    pages = {}
    page_counter = 0

    while queue:
        url = queue.popleft()
        url_norm = normalize_url(url)
        if url_norm in seen:
            continue
        seen.add(url_norm)
        page_counter += 1

        if url_norm.lower().endswith(".pdf"):
            text = extract_text_from_pdf_url(url_norm)
            pages[url_norm] = {"text": text, "title": url_norm}
            ordered_urls.append(url_norm)
            print(f"[{page_counter}] [PDF] {url_norm} (text len: {len(text)})")
            if max_pages and len(ordered_urls) >= max_pages:
                break
            continue

        try:
            r = requests.get(url_norm, headers={"User-Agent": USER_AGENT}, timeout=15)
            r.raise_for_status()
            html = r.text
            text = extract_text_from_html(html)
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else url_norm
            pages[url_norm] = {"text": text, "title": title}
            ordered_urls.append(url_norm)
            print(f"[{page_counter}] [HTML] {url_norm} (text len: {len(text)})")

            # BFS ссылки
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                try:
                    joined = urljoin(url_norm, href)
                    norm = normalize_url(joined)
                    if norm not in seen and is_allowed_url(norm, seed_domains):
                        queue.append(norm)
                except Exception:
                    continue

        except Exception as e:
            print(f"[{page_counter}] [WARN] Failed {url_norm}: {e}")
            pages[url_norm] = {"text": "", "title": ""}
            ordered_urls.append(url_norm)

        if max_pages and len(ordered_urls) >= max_pages:
            print(f"[INFO] Reached max_pages={max_pages}. Stopping crawl.")
            break
        time.sleep(delay)
    return ordered_urls, pages


# ----- FAISS -----
def build_or_update_faiss(ordered_urls, pages, out_dir, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    faiss_dir = os.path.join(out_dir, "faiss_index")
    existing_text_hashes = set()
    db = None

    # Загрузка существующей базы
    if os.path.exists(faiss_dir):
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        for m in getattr(db, "metadatas", []) or []:
            txt = m.get("text", "")
            if txt:
                existing_text_hashes.add(hashlib.sha1(txt.encode("utf-8")).hexdigest())
        print(f"[INFO] Загружена существующая FAISS база с {len(existing_text_hashes)} чанками.")

    all_chunks = []
    all_metadatas = []

    for url in ordered_urls:
        entry = pages.get(url, {"text": "", "title": ""})
        page_text = entry.get("text", "")
        title = entry.get("title", "") or url
        if not page_text or len(page_text.strip()) < 30:
            continue
        chunks = splitter.split_text(page_text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip() or len(chunk.strip()) < 30:
                continue
            uid = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
            if uid in existing_text_hashes:
                continue
            existing_text_hashes.add(uid)
            all_chunks.append(chunk)
            all_metadatas.append({"source": url, "title": title, "chunk_id": i, "text": chunk})

    print(f"[INFO] Всего новых уникальных чанков: {len(all_chunks)}")
    if not all_chunks:
        print("[INFO] Нет новых чанков для добавления. База не обновлена.")
        return db

    print("[INFO] Добавление чанков в FAISS...")
    if db is None:
        db = FAISS.from_texts([], embeddings)
    for chunk, meta in tqdm(zip(all_chunks, all_metadatas), total=len(all_chunks), desc="Adding chunks to FAISS"):
        db.add_texts([chunk], metadatas=[meta])

    os.makedirs(faiss_dir, exist_ok=True)
    print("[INFO] Сохранение FAISS базы...")
    db.save_local(faiss_dir)
    print(f"[OK] FAISS сохранён в {faiss_dir}")
    return db


# ----- Main -----
def main(args):
    urls = []
    if args.seeds and os.path.exists(args.seeds):
        with open(args.seeds, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    print(f"[START] {len(urls)} seeds loaded.")

    # Загружаем существующие URL
    urls_txt_path = os.path.join(args.out, DEFAULT_URLS_FILE)
    existing_urls = []
    if os.path.exists(urls_txt_path):
        with open(urls_txt_path, "r", encoding="utf-8") as f:
            existing_urls = [line.strip() for line in f if line.strip()]

    ordered_urls, pages = crawl(urls, max_pages=args.max_pages, delay=args.delay)

    # Объединяем уникальные URL
    all_urls = existing_urls.copy()
    for u in ordered_urls:
        if u not in all_urls:
            all_urls.append(u)

    os.makedirs(args.out, exist_ok=True)
    with open(urls_txt_path, "w", encoding="utf-8") as f:
        for u in all_urls:
            f.write(u + "\n")
    print(f"[OK] urls.txt обновлён: {len(all_urls)} URL")

    # Обработка локальных PDF
    pdf_dir = Path(args.pdf_path) if args.pdf_path else None
    pdf_files = list(pdf_dir.rglob("*.pdf")) if pdf_dir and pdf_dir.exists() else []

    # LM Embeddings
    embedder = LMStudioEmbeddings(EMBEDDING_MODEL, LM_API, LM_API_KEY)

    # Обновление FAISS
    build_or_update_faiss(
        ordered_urls + [str(f.resolve()) for f in pdf_files],
        {**pages, **{str(f.resolve()): {"text": extract_text_from_pdf_file(f), "title": str(f)} for f in pdf_files}},
        args.out,
        embedder
    )
    print("[DONE] FAISS и urls.txt обновлены.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl pages, process PDFs, update FAISS KB")
    parser.add_argument("--seeds", "-s", help="Seed URLs file")
    parser.add_argument("--pdf_path", "-p", help="Directory with PDF files")
    parser.add_argument("--out", "-o", default=DEFAULT_OUT, help="Output folder")
    parser.add_argument("--max-pages", "-m", type=int, default=None, help="Max pages to crawl")
    parser.add_argument("--delay", "-d", type=float, default=0.2, help="Delay between requests")
    args = parser.parse_args()
    main(args)
