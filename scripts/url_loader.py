#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG FAISS Builder с поддержкой:
- HTML страниц (BFS обход, max_pages)
- PDF файлов (локальные + ссылки на PDF)
- Обновление базы FAISS без дубликатов
- urls.txt с уникальными URL
- Фильтрация URL по seed-доменам, исключение mailto/tel
- Защита от зацикливания на 404 страницах
"""

import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from scripts.model_init import add_chunks_to_faiss, USER_AGENT
from scripts.pdf_loader import extract_text_from_pdf_file


# ---- Настройки ----
DEFAULT_OUT = "kb_output"
DEFAULT_URLS_FILE = "find_urls.txt"
DEFAULT_SEEDS_FILE = "seed_urls.txt"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CONSECUTIVE_WARNS = 5  # Максимум предупреждений подряд перед остановкой

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


# ----- PDF обработка -----
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
    for sd in seed_domains:
        #print(url, sd, url.startswith(sd))
        if url.startswith(sd):
            return True
    return False


# ----- BFS Crawl с защитой от зацикливания на 404 -----
def crawl(seeds, max_pages=None, delay=0.2):
    seed_domains = seeds
    queue = deque(normalize_url(s) for s in seeds)
    seen = set()
    ordered_urls = []
    pages = {}
    page_counter = 0
    consecutive_warns = 0  # Счетчик последовательных предупреждений
    
    # Словарь для отслеживания глубины URL (сколько раз мы поднимались)
    url_depths = {}
    for seed in seeds:
        url_depths[normalize_url(seed)] = 0

    while queue:
        url = queue.popleft()
        url_norm = normalize_url(url)
        if url_norm in seen:
            continue
        seen.add(url_norm)
        page_counter += 1

        # Получаем текущую глубину URL или устанавливаем по умолчанию
        current_depth = url_depths.get(url_norm, 0)

        if url_norm.lower().endswith(".pdf"):
            text = extract_text_from_pdf_url(url_norm)
            pages[url_norm] = {"text": text, "title": url_norm}
            ordered_urls.append(url_norm)
            print(f"[{page_counter}] [PDF] {url_norm} (text len: {len(text)})")
            consecutive_warns = 0  # Сбрасываем счетчик при успешной обработке
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
            consecutive_warns = 0  # Сбрасываем счетчик при успешной обработке

            # BFS ссылки с учетом текущей глубины
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                try:
                    joined = urljoin(url_norm, href)
                    norm = normalize_url(joined)
                    if norm not in seen and is_allowed_url(norm, seed_domains):
                        # Сохраняем глубину для нового URL
                        url_depths[norm] = current_depth
                        queue.append(norm)
                except Exception:
                    continue

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"[{page_counter}] [WARN] Failed {url_norm}: {e}")
                pages[url_norm] = {"text": "", "title": ""}
                ordered_urls.append(url_norm)
                consecutive_warns += 1
                
                # Проверяем, не достигли ли мы лимита предупреждений
                if consecutive_warns >= MAX_CONSECUTIVE_WARNS:
                    print(f"[WARN] Достигнуто {MAX_CONSECUTIVE_WARNS} предупреждений подряд. Поднимаемся на уровень выше...")
                    
                    # Поднимаемся на уровень выше - ищем родительский URL
                    parent_url = get_parent_url(url_norm)
                    if parent_url and parent_url not in seen:
                        print(f"[INFO] Добавляем родительский URL в очередь: {parent_url}")
                        queue.appendleft(parent_url)  # Добавляем в начало очереди
                        url_depths[parent_url] = current_depth + 1
                    
                    consecutive_warns = 0  # Сбрасываем счетчик после подъема
                    
            else:
                # Для других HTTP ошибок просто логируем
                print(f"[{page_counter}] [WARN] Failed {url_norm}: {e}")
                pages[url_norm] = {"text": "", "title": ""}
                ordered_urls.append(url_norm)
                consecutive_warns += 1
                
        except Exception as e:
            print(f"[{page_counter}] [WARN] Failed {url_norm}: {e}")
            pages[url_norm] = {"text": "", "title": ""}
            ordered_urls.append(url_norm)
            consecutive_warns += 1

        # Проверяем общий лимит предупреждений для не-404 ошибок
        if consecutive_warns >= MAX_CONSECUTIVE_WARNS:
            print(f"[WARN] Достигнуто {MAX_CONSECUTIVE_WARNS} ошибок подряд. Пропускаем следующие URL...")
            # Пропускаем следующие URL до тех пор, пока не найдем рабочий
            # Это предотвращает зацикливание на битых ссылках
            
        if max_pages and len(ordered_urls) >= max_pages:
            print(f"[INFO] Reached max_pages={max_pages}. Stopping crawl.")
            break
        time.sleep(delay)
    
    return ordered_urls, pages


def get_parent_url(url):
    """
    Возвращает родительский URL на один уровень выше.
    Например: 
    https://example.com/path/to/page -> https://example.com/path/
    https://example.com/path/ -> https://example.com/
    """
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) > 1:
            # Убираем последнюю часть пути
            new_path = '/' + '/'.join(path_parts[:-1]) + '/'
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                new_path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
        elif len(path_parts) == 1 and path_parts[0]:
            # Поднимаемся до корня
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                '/',
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
        else:
            # Уже в корне, некуда подниматься
            return None
    except Exception:
        return None


def crawl_and_update_faiss(embedder, seeds_file: str, output_dir: str, max_pages: int = 100, delay: float = 0.2, pdf_path: str = None):
    """
    Главная функция для обработки seed URL, обхода HTML и PDF ссылок, обновления FAISS.
    """

    # Загружаем seeds
    seeds = []
    if os.path.exists(seeds_file):
        with open(seeds_file, "r", encoding="utf-8") as f:
            seeds = [line.strip() for line in f if line.strip()]

    print(f"[START] {len(seeds)} seeds loaded.")

    # Обход URL
    ordered_urls, pages = crawl(seeds, max_pages=max_pages, delay=delay)

    # Добавляем локальные PDF, если есть
    pdf_items = {}
    if pdf_path:
        pdf_dir = Path(pdf_path)
        for f in pdf_dir.rglob("*.pdf"):
            text = extract_text_from_pdf_file(f)
            if text.strip():
                pdf_items[str(f.resolve())] = {"text": text, "title": f.stem}

    # Объединяем все данные
    all_items = {**pages, **pdf_items}

    # Обновление FAISS
    add_chunks_to_faiss(all_items, output_dir, embedder)

    # Обновление urls.txt
    urls_txt_path = Path(output_dir) / "find_urls.txt"
    all_urls = list(pages.keys())
    if pdf_items:
        all_urls += list(pdf_items.keys())
    os.makedirs(output_dir, exist_ok=True)
    with open(urls_txt_path, "w", encoding="utf-8") as f:
        for u in all_urls:
            f.write(u + "\n")
    print(f"[DONE] FAISS и urls.txt обновлены ({len(all_urls)} источников).")