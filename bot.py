import asyncio
import logging
import json
from datetime import datetime

from maxapi import Bot, Dispatcher
from maxapi.types import (
    CallbackButton,
    MessageCreated, 
    MessageCallback,
    CommandStart,
    Command
)
from maxapi.utils.inline_keyboard import InlineKeyboardBuilder

from reminders import ReminderManager



# Загружаем данные из JSON файлов
with open('jsons/FAQ.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

with open('categories.json', 'r', encoding='utf-8') as f:
    categories_data = json.load(f)

logging.basicConfig(level=logging.INFO)

bot = Bot('f9LHodD0cOJgDVVnKfwRanQrYXyiuaCq0EdOcsAdfkarSVVmJbZoolSECS7NWJhX_D12PSPLYDrjw_fqbq2v')
dp = Dispatcher()

# Добавьте эту проверку в начало вашего кода
print("Available bot methods:", [method for method in dir(bot) if not method.startswith('_')])

# Инициализируем менеджер напоминаний
reminder_manager = ReminderManager(bot)

# Словарь для хранения состояний пользователей
user_modes = {}

# Функция нормализации строк
def normalize_string(s):
    """Нормализация строки для сравнения - убираем лишние пробелы"""
    return ' '.join(s.strip().split())

# Нормализуем FAQ данные
normalized_faq_data = {}
for question, answer in faq_data.items():
    normalized_question = normalize_string(question)
    normalized_faq_data[normalized_question] = answer

# Нормализуем категории
normalized_categories_data = {}
for category, data in categories_data.items():
    normalized_questions = [normalize_string(q) for q in data.get("questions", [])]
    normalized_categories_data[category] = {
        "title": data.get("title", ""),
        "questions": normalized_questions
    }

# Функция для получения вопросов по категории
def get_questions_for_category(category):
    return normalized_categories_data.get(category, {}).get("questions", [])

# Функция для получения заголовка категории
def get_category_title(category):
    return normalized_categories_data.get(category, {}).get("title", "Категория")

# Функция для получения оригинальных вопросов по категории (для отображения)
def get_original_questions_for_category(category):
    return categories_data.get(category, {}).get("questions", [])


#============================================================================
# Инициализация ИИ
#============================================================================
from scripts.model_init import get_embedder
from scripts.rag import init_bot, init_bot2, qa_ai, qa_ai_nav, PROMPT1, PROMPT2

embedder = get_embedder()
DEFAULT_OUT = "kb_output"
qa_chain = init_bot(embedder, DEFAULT_OUT, prompt=PROMPT1)
qa_chain_map = init_bot2(prompt=PROMPT2)

import re
import logging
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
async def find_navigation_images(answer: str) -> list[str]:
    """
    Извлекает путь к зданию и номер аудитории из ответа,
    возвращает список GitHub URLs к изображениям, которые существуют.
    """
    logging.info("зашли в find_navigation_images")
    
    match = re.search(r"на рисунке\s*['\"]([^'\"]+)['\"]", answer, re.IGNORECASE)
    if not match:
        return []

    path_str = match.group(1).strip().replace('\\', '/')
    parts = path_str.split('/', 1)
    if len(parts) != 2:
        return []

    building, filename = parts
    rm = re.search(r'(\d+)', filename)
    if not rm:
        return []

    room = rm.group(1)
    
    # Базовый URL вашего GitHub репозитория
    base_url = "https://raw.githubusercontent.com/Ogneva2Vasilisa/Min_RAG_with_crawler/main/img"
    
    # Правильно кодируем кириллицу в URL
    building_encoded = urllib.parse.quote(building)
    
    # Формируем возможные URL с кодированием
    possible_urls = [
        f"{base_url}/{building_encoded}/{room}_1.jpg",
        f"{base_url}/{building_encoded}/{room}_2.jpg", 
    ]
    
    # Убираем возможные двойные слеши
    clean_urls = [url.replace('//', '/').replace(':/', '://') for url in possible_urls]
    
    # Проверяем существование файлов
    existing_urls = []
    for url in clean_urls:
        try:
            # Создаем запрос и проверяем статус
            req = urllib.request.Request(url, method='HEAD')
            response = urllib.request.urlopen(req)
            if response.status == 200:
                existing_urls.append(url)
                logging.info(f"Файл существует: {url}")
        except (HTTPError, URLError) as e:
            logging.info(f"Файл не существует: {url} - {e}")
            continue
        except Exception as e:
            logging.warning(f"Ошибка при проверке {url}: {e}")
            continue
    
    logging.info(f"Найдено существующих файлов: {len(existing_urls)}")
    return existing_urls

async def send_navigation_response(event, answer: str):
    """
    Упрощенная отправка навигации - текст и затем ссылки на изображения
    """
    logging.info("зашли в send_navigation_response")
    
    # Очищаем текст ответа
    nav_text = answer.split('\n')[0].strip()
    
    # Получаем URL изображений (только существующие)
    image_urls = await find_navigation_images(answer)

    if not image_urls:
        await event.message.answer("❌ Сейчас данное место не поддерживается ботом или его не существует")
        await event.message.answer("Для выхода из режима используйте /cancel")
        return

    # Отправляем каждую существующую ссылку отдельным сообщением
    for i, url in enumerate(image_urls, 1):
        await event.message.answer(f"📸 Схема {i}: {url}")

    await event.message.answer("Для выхода из режима используйте /cancel")

# ============================================================================
# ОБНОВЛЕННЫЕ ФУНКЦИИ МЕНЮ
# ============================================================================

# НОВОЕ ГЛАВНОЕ МЕНЮ с 5 кнопками
def get_main_menu():
    builder = InlineKeyboardBuilder()
    
    # 5 основных кнопок
    builder.row(CallbackButton(text="📅 Напоминания", payload="reminders_menu"))
    builder.row(CallbackButton(text="❓ Часто задаваемые вопросы", payload="faq_categories"))
    builder.row(CallbackButton(text="💬 Задать свободный вопрос", payload="free_question"))
    builder.row(CallbackButton(text="🗺️ Навигация по университету", payload="navigation"))
    builder.row(CallbackButton(text="ℹ️ Помощь по боту", payload="bot_help"))
    
    return builder.as_markup()

# Меню категорий FAQ (вместо старого главного меню)
def get_faq_categories_menu():
    builder = InlineKeyboardBuilder()
    
    # Добавляем кнопки по категориям FAQ
    builder.row(CallbackButton(text="🌱 Адаптация первокурсников", payload="menu_freshmen"))
    builder.row(CallbackButton(text="📚 Учебный процесс", payload="menu_studies"))
    builder.row(CallbackButton(text="📄 Документы и справки", payload="menu_documents"))
    builder.row(CallbackButton(text="🎉 Студенческая жизнь", payload="menu_campus_life"))
    builder.row(CallbackButton(text="🔧 Техническая поддержка", payload="menu_support"))
    builder.row(CallbackButton(text="💳 Финансовые вопросы", payload="menu_finance"))
    builder.row(CallbackButton(text="🔬 Наука и исследования", payload="menu_research"))
    builder.row(CallbackButton(text="🔙 Назад в главное меню", payload="back_to_main"))
    
    return builder.as_markup()

# Функция для меню напоминаний
def get_reminders_menu():
    builder = InlineKeyboardBuilder()
    builder.row(CallbackButton(text="➕ Добавить напоминание", payload="add_reminder"))
    builder.row(CallbackButton(text="📅 Напоминания на эту неделю", payload="week_reminders"))
    builder.row(CallbackButton(text="✏️ Изменить/удалить по дате", payload="edit_by_date"))
    builder.row(CallbackButton(text="🔙 Назад в главное меню", payload="back_to_main"))
    return builder.as_markup()

# Функция для создания меню вопросов по категории
def get_questions_menu(category):
    builder = InlineKeyboardBuilder()
    
    # Используем оригинальные вопросы для отображения
    original_questions = get_original_questions_for_category(category)
    
    for question in original_questions:
        category_simple = category.replace("menu_", "")
        question_index = original_questions.index(question)
        question_id = f"q_{category_simple}_{question_index}"
        builder.row(CallbackButton(text=question, payload=question_id))
    
    builder.row(CallbackButton(text="🔙 Назад к категориям", payload="back_to_faq_categories"))
    
    return builder.as_markup()

# Функция для меню управления напоминаниями на неделю
def get_week_reminders_menu(reminders):
    builder = InlineKeyboardBuilder()
    
    for reminder_id, text, date_str in reminders:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        display_text = f"{date.strftime('%d.%m')}: {text[:25]}{'...' if len(text) > 25 else ''}"
        
        # Кнопка удаления с реальным ID
        builder.row(CallbackButton(text=f"❌ ID {reminder_id}: {display_text}", payload=f"delete_{reminder_id}"))
    
    builder.row(CallbackButton(text="🔙 Назад к напоминаниям", payload="back_to_reminders"))
    return builder.as_markup()

# Функция для меню управления напоминаниями на конкретную дату
def get_date_reminders_menu(reminders, target_date):
    builder = InlineKeyboardBuilder()
    
    for i, (reminder_id, text, date_str) in enumerate(reminders, 1):
        display_text = f"{text[:30]}{'...' if len(text) > 30 else ''}"
        # Показываем порядковый номер для пользователя, но используем реальный ID в payload
        builder.row(CallbackButton(text=f"#{i} ✏️ {display_text}", payload=f"edit_text_{reminder_id}"))
        builder.row(CallbackButton(text=f"#{i} ❌ Удалить", payload=f"delete_{reminder_id}"))
    
    builder.row(CallbackButton(text="🔙 Назад к напоминаниям", payload="back_to_reminders"))
    return builder.as_markup()

# Функция для получения ответа на вопрос
def get_answer(question):
    """Получение ответа на вопрос с нормализацией строки"""
    normalized_question = normalize_string(question)
    
    logging.info(f"Поиск ответа для: '{question}' -> нормализовано: '{normalized_question}'")
    
    # Прямой поиск в нормализованных данных
    if normalized_question in normalized_faq_data:
        logging.info(f"Найден прямой ответ для: '{normalized_question}'")
        return normalized_faq_data[normalized_question]
    
    # Если не нашли - дополнительный поиск
    for faq_question, answer in normalized_faq_data.items():
        if normalized_question == faq_question:
            logging.info(f"Найден ответ при дополнительном поиске: '{faq_question}'")
            return answer
    
    # Частичное совпадение как запасной вариант
    for faq_question, answer in normalized_faq_data.items():
        if normalized_question in faq_question or faq_question in normalized_question:
            logging.info(f"Найден ответ при частичном совпадении: '{faq_question}'")
            return answer
    
    logging.warning(f"Ответ не найден для: '{normalized_question}'")
    available_keys = list(normalized_faq_data.keys())
    if available_keys:
        logging.warning(f"Доступные ключи в FAQ (первые 5): {available_keys[:5]}")
    
    return "Ответ на данный вопрос временно недоступен."


# ============================================================================
# ОБРАБОТЧИКИ КОМАНД
# ============================================================================

# Обработчик команды /start
@dp.message_created(CommandStart())
async def send_welcome(event: MessageCreated):
    # Сбрасываем режим пользователя при старте
    chat_id = event.message.recipient.chat_id
    user_modes[chat_id] = None
    
    welcome_text = (
        "👋 Добро пожаловать в студенческий помощник Политеха!\n\n"
        "Я помогу вам с учебными вопросами, напоминаниями и навигацией по университету.\n\n"
        "Выберите нужный раздел:"
    )
    await event.message.answer(welcome_text, attachments=[get_main_menu()])

# Обработчик команды /menu
@dp.message_created(Command('menu'))
async def show_menu(event: MessageCreated):
    # Сбрасываем режим пользователя при возврате в меню
    chat_id = event.message.recipient.chat_id
    user_modes[chat_id] = None
    
    await event.message.answer("Главное меню:", attachments=[get_main_menu()])

# Обработчик команды /cancel для выхода из режимов
@dp.message_created(Command('cancel'))
async def cancel_mode(event: MessageCreated):
    chat_id = event.message.recipient.chat_id
    current_mode = user_modes.get(chat_id)
    
    if current_mode == 'free_question':
        user_modes[chat_id] = None
        await event.message.answer(
            "✅ Вы вышли из режима свободного вопроса.",
            attachments=[get_main_menu()]
        )
    elif current_mode == 'navigation':
        user_modes[chat_id] = None
        await event.message.answer(
            "✅ Вы вышли из режима навигации.",
            attachments=[get_main_menu()]
        )
    else:
        await event.message.answer(
            "Вы не находитесь в специальном режиме.",
            attachments=[get_main_menu()]
        )

# Обработчик команды /remind
@dp.message_created(Command('remind'))
async def set_reminder_command(event: MessageCreated):
    # Сбрасываем режим пользователя при использовании команды напоминания
    chat_id = event.message.recipient.chat_id
    user_modes[chat_id] = None
    
    try:
        parts = event.message.body.text.split(' ', 2)
        if len(parts) < 3:
            await event.message.answer(
                "Формат: /remind ДД.ММ.ГГГГ текст напоминания\n\nНапример:\n/remind 25.12.2024 Новогодний ужин",
                attachments=[get_reminders_menu()]
            )
            return

        date_str = parts[1]
        text = parts[2]

        event_date = datetime.strptime(date_str, '%d.%m.%Y').date()
        today = datetime.now().date()
        
        if event_date <= today:
            await event.message.answer(
                "Дата должна быть в будущем!",
                attachments=[get_reminders_menu()]
            )
            return

        # Правильное получение chat_id для Max API
        chat_id = event.message.recipient.chat_id
        
        reminder_id = await reminder_manager.add_reminder(chat_id, text, event_date)

        await event.message.answer(
            f"✅ Напоминание установлено! (ID: {reminder_id})\n"
            f"Событие: {text}\n"
            f"Дата: {event_date.strftime('%d.%m.%Y')}\n\n"
            f"Вы получите напоминания:\n"
            f"• Вечером накануне в 18:00\n"
            f"• Утром в день события в 9:00",
            attachments=[get_reminders_menu()]
        )

    except ValueError:
        await event.message.answer(
            "Неверный формат даты. Используйте: ДД.ММ.ГГГГ",
            attachments=[get_reminders_menu()]
        )
    except Exception as e:
        await event.message.answer(
            "Произошла ошибка. Попробуйте еще раз.",
            attachments=[get_reminders_menu()]
        )
        logging.error(f"Ошибка установки напоминания: {e}")

# ============================================================================
# ОБНОВЛЕННЫЙ ОСНОВНОЙ ОБРАБОТЧИК КНОПОК
# ============================================================================

@dp.message_callback()
async def handle_button_click(callback: MessageCallback):
    if hasattr(callback, 'callback') and hasattr(callback.callback, 'payload'):
        payload = callback.callback.payload
    else:
        try:
            callback_data = callback.model_dump()
            payload = callback_data.get('callback', {}).get('payload')
        except:
            payload = None
    
    if not payload:
        await callback.message.answer("Ошибка обработки запроса.", attachments=[get_main_menu()])
        return

    print("Extracted payload:", payload)
    
    # ПРАВИЛЬНОЕ получение chat_id для Max API в callback
    chat_id = callback.message.recipient.chat_id
    
    # Обработка кнопки "Назад" в главное меню
    if payload == "back_to_main":
        user_modes[chat_id] = None  # Сбрасываем режим
        await callback.message.answer("Главное меню:", attachments=[get_main_menu()])
        return
    
    # Обработка кнопки "Назад" к категориям FAQ
    if payload == "back_to_faq_categories":
        user_modes[chat_id] = None  # Сбрасываем режим
        await callback.message.answer("❓ Часто задаваемые вопросы:", attachments=[get_faq_categories_menu()])
        return
    
    # Обработка кнопки "Назад" в меню напоминаний
    if payload == "back_to_reminders":
        user_modes[chat_id] = None  # Сбрасываем режим
        await callback.message.answer("📅 Управление напоминаниями:", attachments=[get_reminders_menu()])
        return
    
    # Обработка главного меню напоминаний
    if payload == "reminders_menu":
        user_modes[chat_id] = None  # Сбрасываем режим
        await callback.message.answer("📅 Управление напоминаниями:", attachments=[get_reminders_menu()])
        return
    
    # Обработка кнопки FAQ категорий
    if payload == "faq_categories":
        user_modes[chat_id] = None  # Сбрасываем режим
        await callback.message.answer(
            "❓ Выберите категорию часто задаваемых вопросов:",
            attachments=[get_faq_categories_menu()]
        )
        return
    
    # Обработка свободного вопроса
    if payload == "free_question":
        user_modes[chat_id] = 'free_question'
        await callback.message.answer(
            "⏳ Подождите, пока система обработает запрос...\n\n"
            "✅ Система готова! Задайте ваш вопрос.\n\n"
            "💡 *Режим свободного вопроса активирован*\n"
            "Для выхода используйте команду /cancel",
            attachments=None
        )
        return
    
    # Обработка навигации
    if payload == "navigation":
        user_modes[chat_id] = 'navigation'
        await callback.message.answer(
            "⏳ Подождите, пока система обработает запрос...\n\n"
            "✅ Система готова! Введите ваш навигационный запрос.\n\n"
            "🗺️ *Режим навигации активирован*\n"
            "Для выхода используйте команду /cancel",
            attachments=None
        )
        return
    
    # Обработка помощи по боту (заглушка)
    if payload == "bot_help":
        user_modes[chat_id] = None
        await callback.message.answer(
            "ℹ️ Помощь по боту:\n\n"
            "📅 **Напоминания** - устанавливайте напоминания о важных событиях\n"
            "❓ **Часто задаваемые вопросы** - ответы на популярные вопросы студентов\n"
            "💬 **Свободный вопрос** - задайте любой вопрос (режим эхо-ответа)\n"
            "🗺️ **Навигация** - найдите нужное место в университете (режим эхо-ответа)\n\n"
            "Команды:\n"
            "/menu - главное меню\n"
            "/cancel - выход из режимов\n"
            "/remind ДД.ММ.ГГГГ текст - установить напоминание\n"
            "/edit_text ID новый_текст - изменить текст напоминания",
            attachments=[get_main_menu()]
        )
        return
    
    # Обработка добавления напоминания
    if payload == "add_reminder":
        user_modes[chat_id] = None
        await callback.message.answer(
            "Чтобы установить напоминание, отправьте команду:\n"
            "/remind ДД.ММ.ГГГГ текст напоминания\n\n"
            "Например:\n"
            "/remind 25.12.2024 Новогодний ужин",
            attachments=[get_reminders_menu()]
        )
        return
    
    # Обработка показа напоминаний на неделю
    if payload == "week_reminders":
        user_modes[chat_id] = None
        reminders = await reminder_manager.get_week_reminders(chat_id)
        
        if not reminders:
            await callback.message.answer(
                "На эту неделю у вас нет напоминаний.",
                attachments=[get_reminders_menu()]
            )
            return
        
        message = "📅 Ваши напоминания на эту неделю:\n\n"
        for reminder_id, text, date_str in reminders:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            message += f"• ID {reminder_id}: {date.strftime('%d.%m.%Y')} - {text}\n"
        
        message += "\nДля редактирования используйте команду: /edit_text [ID] [новый_текст]"
        
        await callback.message.answer(
            message,
            attachments=[get_week_reminders_menu(reminders)]
        )
        return
    
    # Обработка изменения/удаления по дате
    if payload == "edit_by_date":
        user_modes[chat_id] = None
        await callback.message.answer(
            "Введите дату в формате ДД.ММ.ГГГГ для просмотра напоминаний:\n"
            "Например: 25.12.2024",
            attachments=[get_reminders_menu()]
        )
        return
    
    # Обработка редактирования текста напоминания
    if payload.startswith("edit_text_"):
        user_modes[chat_id] = None
        try:
            reminder_id = int(payload.split("_")[2])
            # Получаем информацию о напоминании
            reminders = await reminder_manager.get_user_reminders(chat_id)
            target_reminder = None
            for rem_id, text, date_str in reminders:
                if rem_id == reminder_id:
                    target_reminder = (rem_id, text, date_str)
                    break
            
            if target_reminder:
                rem_id, text, date_str = target_reminder
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                await callback.message.answer(
                    f"✏️ Редактирование напоминания (ID: {reminder_id}):\n\n"
                    f"Текущий текст: {text}\n"
                    f"Дата: {date.strftime('%d.%m.%Y')}\n\n"
                    f"Для изменения текста отправьте:\n"
                    f"/edit_text {reminder_id} новый_текст\n\n"
                    f"Например:\n"
                    f"/edit_text {reminder_id} Встреча с деканом в 15:00",
                    attachments=[get_reminders_menu()]
                )
            else:
                await callback.message.answer(
                    "Напоминание не найдено.",
                    attachments=[get_reminders_menu()]
                )
        except Exception as e:
            await callback.message.answer(
                "Ошибка при редактировании напоминания.",
                attachments=[get_reminders_menu()]
            )
        return
    
    # Обработка удаления напоминания
    if payload.startswith("delete_"):
        user_modes[chat_id] = None
        try:
            reminder_id = int(payload.split("_")[1])
            success = await reminder_manager.delete_reminder(reminder_id, chat_id)
            if success:
                await callback.message.answer(
                    f"✅ Напоминание (ID: {reminder_id}) удалено!",
                    attachments=[get_reminders_menu()]
                )
            else:
                await callback.message.answer(
                    f"❌ Не удалось удалить напоминание (ID: {reminder_id})",
                    attachments=[get_reminders_menu()]
                )
        except Exception as e:
            await callback.message.answer(
                "❌ Ошибка при удалении напоминания",
                attachments=[get_reminders_menu()]
            )
        return
    
    # Обработка вопросов FAQ
    if payload.startswith("q_"):
        user_modes[chat_id] = None
        parts = payload.split("_")
        if len(parts) >= 3:
            category_simple = parts[1]
            category = f"menu_{category_simple}"
            
            try:
                question_index = int(parts[2])
                # Получаем нормализованный вопрос для поиска ответа
                normalized_questions = get_questions_for_category(category)
                # Получаем оригинальный вопрос для отображения
                original_questions = get_original_questions_for_category(category)
                
                if 0 <= question_index < len(normalized_questions):
                    normalized_question = normalized_questions[question_index]
                    original_question = original_questions[question_index] if question_index < len(original_questions) else normalized_question
                    
                    answer = get_answer(normalized_question)
                    await callback.message.answer(
                        f"**{original_question}**\n\n{answer}",
                        attachments=[get_questions_menu(category)]
                    )
                    return
            except (ValueError, IndexError) as e:
                logging.error(f"Ошибка обработки вопроса: {e}")
                pass
    
    # Обработка категорий FAQ (старые категории теперь в меню FAQ)
    if payload in categories_data:
        user_modes[chat_id] = None
        category_title = get_category_title(payload)
        await callback.message.answer(
            f"{category_title}\n\nВыберите интересующий вас вопрос:",
            attachments=[get_questions_menu(payload)]
        )
    else:
        user_modes[chat_id] = None
        await callback.message.answer(
            "Извините, раздел временно недоступен.",
            attachments=[get_main_menu()]
        )

# ============================================================================
# ОБРАБОТЧИКИ СООБЩЕНИЙ ДЛЯ РЕДАКТИРОВАНИЯ
# ============================================================================

# Команда для редактирования текста напоминания
@dp.message_created(Command('edit_text'))
async def edit_text_reminder_command(event: MessageCreated):
    # Сбрасываем режим пользователя при использовании команды редактирования
    chat_id = event.message.recipient.chat_id
    user_modes[chat_id] = None
    
    try:
        parts = event.message.body.text.split(' ', 2)
        if len(parts) < 3:
            await event.message.answer(
                "Формат: /edit_text ID новый_текст\n\nНапример:\n/edit_text 5 Встреча с деканом в 15:00\n\nID напоминания можно посмотреть в списке напоминаний по дате (в скобках после #).",
                attachments=[get_reminders_menu()]
            )
            return

        reminder_id = int(parts[1])
        new_text = parts[2]

        # Правильное получение chat_id для Max API
        chat_id = event.message.recipient.chat_id
        
        success = await reminder_manager.update_reminder_text(reminder_id, chat_id, new_text)
        
        if success:
            # После успешного обновления показываем обновленный список напоминаний на неделю
            reminders = await reminder_manager.get_week_reminders(chat_id)
            
            if not reminders:
                await event.message.answer(
                    f"✅ Текст напоминания (ID: {reminder_id}) обновлен!\n"
                    f"Новый текст: {new_text}\n\n"
                    f"На эту неделю у вас нет напоминаний.",
                    attachments=[get_reminders_menu()]
                )
                return
            
            message = f"✅ Текст напоминания (ID: {reminder_id}) обновлен!\nНовый текст: {new_text}\n\n📅 Ваши напоминания на эту неделю:\n\n"
            for rem_id, text, date_str in reminders:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                message += f"• ID {rem_id}: {date.strftime('%d.%m.%Y')} - {text}\n"
            
            await event.message.answer(
                message,
                attachments=[get_week_reminders_menu(reminders)]
            )
        else:
            # Если не удалось обновить, показываем список напоминаний с реальными ID
            reminders = await reminder_manager.get_user_reminders(chat_id)
            if reminders:
                message = f"❌ Не удалось обновить текст напоминания (ID: {reminder_id}).\n\n"
                message += "📅 Ваши напоминания (с реальными ID):\n\n"
                for rem_id, text, date_str in reminders:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    message += f"• ID {rem_id}: {date.strftime('%d.%m.%Y')} - {text}\n"
                
                message += f"\nИспользуйте правильный ID из списка выше."
                await event.message.answer(message, attachments=[get_reminders_menu()])
            else:
                await event.message.answer(
                    f"❌ Не удалось обновить текст напоминания (ID: {reminder_id}). У вас нет напоминаний.",
                    attachments=[get_reminders_menu()]
                )

    except ValueError as e:
        await event.message.answer(
            "Ошибка формата. Используйте: /edit_text ID новый_текст\n\nID - это номер напоминания из базы данных (показан в скобках в списке напоминаний по дате).",
            attachments=[get_reminders_menu()]
        )
    except Exception as e:
        await event.message.answer(
            "Произошла ошибка при редактировании.",
            attachments=[get_reminders_menu()]
        )
        logging.error(f"Ошибка редактирования текста напоминания: {e}")

# Обработчик для ввода даты при редактировании по дате
@dp.message_created()
async def handle_date_input(event: MessageCreated):
    # Проверяем, является ли сообщение датой в формате ДД.ММ.ГГГГ ???
    text = event.message.body.text.strip()
    print(text)
    print(type(text))
    # Проверяем режим пользователя
    chat_id = event.message.recipient.chat_id
    current_mode = user_modes.get(chat_id)
    # 1) Режим "свободный вопрос" — сначала сообщение-плейсхолдер, потом вычисление ответа и отправка
    if current_mode == 'free_question':
        # Сообщаем пользователю, что запрос обрабатывается
        await event.message.answer("⏳ Подождите, ваш вопрос обрабатывается...", attachments=None)

        try:
            
            answer, s = qa_ai(qa_chain, text)
        except Exception as e:
            logging.error(f"Ошибка при генерации ответа (free_question): {e}")
            await event.message.answer(
                "❌ Произошла ошибка при обработке запроса. Попробуйте ещё раз или используйте /cancel.",
                attachments=[get_main_menu()]
            )
            return
        if answer=='':
            await event.message.answer(
            f"🔍 *Ответ (в разработке):* \nСейчас я не могу ответить на Ваш вопрос, попробуйте спросить что-то другое :)\n\n"
            f"Для выхода из режима используйте /cancel"
        )
        else:
            await event.message.answer(
                f"🔍 *Ответ (в разработке):* {answer}\n\n"
                f"🔍 *📚 Источники (в разработке):* {s}\n\n"
                f"Для выхода из режима используйте /cancel"
            )
        return

    # 2) Режим "навигация" — тоже сначала уведомление, потом поиск + отправка (вместе с картинками)
    if current_mode == 'navigation':
        await event.message.answer("⏳ Подождите, ваш навигационный запрос обрабатывается...", attachments=None)

        try:
            qa_chain_map = init_bot2(prompt=PROMPT2)
            answer = qa_ai_nav(qa_chain_map, text)
        except Exception as e:
            logging.error(f"Ошибка при генерации ответа (navigation): {e}")
            await event.message.answer(
                "❌ Произошла ошибка при поиске навигации. Попробуйте ещё раз или используйте /cancel.",
                attachments=[get_main_menu()]
            )
            return

        # отправляет текст + возможные картинки (внутри функции уже есть fallback-сообщения)
        await send_navigation_response(event, answer)
        logging.info("Где картинка?")
        return
    
    # Если не в специальном режиме, обрабатываем как дату
    try:
        # Пытаемся распарсить дату
        target_date = datetime.strptime(text, '%d.%m.%Y').date()
        today = datetime.now().date()
        
        if target_date < today:
            await event.message.answer(
                "Дата должна быть сегодня или в будущем!",
                attachments=[get_reminders_menu()]
            )
            return
        
        # Правильное получение chat_id для Max API
        chat_id = event.message.recipient.chat_id
        
        # Получаем напоминания на эту дату
        reminders = await reminder_manager.get_reminders_by_date(chat_id, target_date)
        
        if not reminders:
            await event.message.answer(
                f"На {target_date.strftime('%d.%m.%Y')} у вас нет напоминаний.",
                attachments=[get_reminders_menu()]
            )
            return
        
        message = f"📅 Напоминания на {target_date.strftime('%d.%m.%Y')}:\n\n"
        # Добавляем нумерацию для понятности и показываем реальные ID
        for i, (reminder_id, reminder_text, date_str) in enumerate(reminders, 1):
            message += f"#{i} (ID: {reminder_id}): {reminder_text}\n"
        
        message += f"\nДля редактирования используйте команду:\n/edit_text [ID] [новый_текст]\n\n"
        if reminders:
            message += f"Например:\n/edit_text {reminders[0][0]} Новая встреча в 14:00"
        
        await event.message.answer(
            message,
            attachments=[get_date_reminders_menu(reminders, target_date)]
        )
        
    except ValueError:
        # Если не дата и не специальный режим, игнорируем (это может быть обычное сообщение)
        pass

async def main():
    # Инициализируем базу данных напоминаний
    await reminder_manager.init_db()
    
    # Запускаем фоновую задачу для напоминаний
    asyncio.create_task(reminder_manager.send_scheduled_reminders())
    
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())