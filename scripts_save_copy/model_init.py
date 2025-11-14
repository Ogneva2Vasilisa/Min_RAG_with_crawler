#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инициализация модели и эмбеддингов для проекта RAG
"""

import os
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS

USER_AGENT = os.environ.get("USER_AGENT", "rag-crawler/1.0")

# LM Studio настройки
LM_API_URL = "http://localhost:1234/v1"
LM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "Qwen2.5-3B-Instruct"
EMBEDDING_MODEL_NAME = "text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf"


class LMStudioEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, api_url=LM_API_URL, api_key=LM_API_KEY):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}

    def embed_documents(self, texts):
        import requests
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


def get_embedder():
    return LMStudioEmbeddings()


def get_llm():
    return OpenAI(
        openai_api_base=LM_API_URL,
        openai_api_key=LM_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )
