#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from rag_faiss_builder import LMStudioEmbeddings, FAISS_PATH, META_PATH

API_URL = "http://localhost:1234/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"
API_KEY = "lm-studio"
EMBEDDING_MODEL_NAME = "text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf"

# === Инициализация эмбеддингов ===
embeddings = LMStudioEmbeddings(EMBEDDING_MODEL_NAME, API_URL, API_KEY)

# === Загружаем FAISS базу ===
db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === Кастомный prompt для русского языка === используя только предоставленные данные из контекста
PROMPT = """
Ты эксперт по Санкт-Петербургскому политеху.
Ответь на вопрос максимально подробно на русском языке.
Если информации недостаточно, скажи "Информации недостаточно", но не придумывай. Не добавляй ничего лишнего, в ответе должен быть небольшой абзац ответа.
Не добавляй ничего лишнего, в ответе должен быть 1 небольшой абзац текста без дублирующихся предложений и лишних символов.

Контекст:
{context}

Вопрос: {question}
Ответ на русском:
"""
prompt_template = PromptTemplate(input_variables=["context", "question"], template=PROMPT)

# === Подключаем LLM ===
llm = OpenAI(
    openai_api_base=API_URL,
    openai_api_key=API_KEY,
    model_name=MODEL_NAME,
    temperature=0.2
)

# === Создаём RAG-цепочку ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# === Чат ===
print("RAG-бот запущен! Задавай вопросы о кампусе.\n")
while True:
    query = input("❓ Вопрос: ").strip()
    if query.lower() in ["exit", "выход", "quit"]:
        print("Выход из чата.")
        break

    result = qa_chain.invoke(query)
    print("\n🧠 Ответ модели:")
    print(result["result"])
    print("\n📚 Использованные источники:")
    for doc in result["source_documents"]:
        print("-", doc.page_content[:200], "...")
    print("\n" + "-" * 50 + "\n")
