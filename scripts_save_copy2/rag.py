#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from scripts.model_init import get_llm, get_faiss_path
from pathlib import Path

DEFAULT_KB_PATH = "kb_output"  # Путь к FAISS базе
DEFAULT_TOP_K = 3               # Сколько документов возвращать

def start_rag_bot(embeddings, kb_path: str = DEFAULT_KB_PATH, top_k: int = DEFAULT_TOP_K):
    """Запуск RAG-бота с подключением к FAISS"""
    print(f"[INFO] Инициализация эмбеддингов и загрузка FAISS из {kb_path}...")
    
    faiss_path = Path(get_faiss_path(kb_path))
    if not faiss_path.exists():
        print(f"[ERROR] FAISS база не найдена по пути: {faiss_path}")
        return

    db = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # Prompt для работы только с предоставленным контекстом
    PROMPT = """
Ты эксперт по Санкт-Петербургскому политеху.
Ответь на вопрос максимально подробно на русском языке.
Если информации недостаточно, скажи "Информации недостаточно", но не придумывай.
Не добавляй ничего лишнего, в ответе должен быть 1 небольшой абзац текста без дублирующихся предложений.

Контекст:
{context}

Вопрос: {question}
Ответ на русском:
"""
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=PROMPT)

    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    print("RAG-бот запущен! Задавай вопросы о кампусе (для выхода введите exit/выход/quit).\n")
    
    while True:
        query = input("❓ Вопрос: ").strip()
        if query.lower() in ["exit", "выход", "quit"]:
            print("Выход из чата.")
            break

        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        print("\n🧠 Ответ модели:")
        print(answer or "Информации недостаточно")

        if sources:
            print("\n📚 Использованные источники:")
            for doc in sources:
                meta = getattr(doc, "metadata", {})
                source = meta.get("source", "Неизвестно")
                title = meta.get("title", "")
                print(f"- {title} ({source})")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    start_rag_bot()
