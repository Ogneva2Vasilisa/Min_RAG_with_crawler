# -*- coding: utf-8 -*-
"""
Загрузчик JSON-файлов в FAISS базу знаний.
Каждый JSON может содержать либо один объект, либо список объектов.
"""
import json
from pathlib import Path
from scripts.model_init import add_chunks_to_faiss


def load_json_content(json_path: Path) -> str:
    """
    Преобразует JSON в текст для индексации.
    Сохраняет ключевую структуру и значения без потери смысла.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {json_path}: {e}")
        return ""

    def flatten_json(obj, indent=0) -> str:
        """
        Рекурсивно превращает структуру JSON в читаемый текст.
        """
        txt_lines = []
        prefix = "  " * indent

        if isinstance(obj, dict):
            for k, v in obj.items():
                txt_lines.append(f"{prefix}{k}:")
                txt_lines.append(flatten_json(v, indent + 1))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                txt_lines.append(f"{prefix}- {flatten_json(item, indent + 1)}")
        else:
            txt_lines.append(f"{prefix}{str(obj)}")

        return "\n".join(txt_lines)

    return flatten_json(data)


def add_jsons_to_faiss_main(json_dir: str, output_dir: str, embedder):
    """
    Главная функция для добавления всех JSON из указанной папки в FAISS.
    """
    json_dir = Path(json_dir)
    items = {}

    for json_file in json_dir.rglob("*.json"):
        text = load_json_content(json_file)
        if text.strip():
            items[str(json_file.resolve())] = {
                "text": text,
                "title": json_file.stem
            }

    if not items:
        print("[INFO] JSON-файлов для добавления не найдено.")
        return

    add_chunks_to_faiss(items, output_dir, embedder)



#================================================================================

def format_curators_json(input_json: str, output_json: str):
    input_path = Path(input_json)
    output_path = Path(output_json)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) >= 2:
            name, link = value[0], value[1]
            group_number = key.split("/")[-1]  # Берем часть после /
            new_key = f"Куратор группы {group_number} --"
            formatted[new_key] = f"Твой куратор {name}, можешь связаться с ним через {link}"
        else:
            formatted[key] = "Некорректные данные о кураторе"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Сохранено в {output_path}")
