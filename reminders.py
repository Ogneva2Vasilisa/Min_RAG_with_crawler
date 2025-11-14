import asyncio
import aiosqlite
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

class ReminderManager:
    def __init__(self, bot):
        self.bot = bot
        self.db_path = 'reminders.db'
        
    async def init_db(self):
        """Инициализация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    reminder_text TEXT NOT NULL,
                    event_date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.commit()
        logger.info("База данных напоминаний инициализирована")

    async def add_reminder(self, chat_id, text, event_date):
        """Добавление напоминания"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO reminders (chat_id, reminder_text, event_date) VALUES (?, ?, ?)",
                (chat_id, text, event_date)
            )
            reminder_id = cursor.lastrowid
            await db.commit()
        logger.info(f"Добавлено напоминание ID {reminder_id} для chat_id {chat_id}")
        return reminder_id

    async def get_user_reminders(self, chat_id):
        """Получение всех напоминаний пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, reminder_text, event_date FROM reminders WHERE chat_id = ? ORDER BY event_date",
                (chat_id,)
            ) as cursor:
                reminders = await cursor.fetchall()
        return reminders

    async def get_week_reminders(self, chat_id):
        """Получение напоминаний на текущую неделю"""
        today = datetime.now().date()
        week_end = today + timedelta(days=7)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, reminder_text, event_date FROM reminders WHERE chat_id = ? AND event_date BETWEEN ? AND ? ORDER BY event_date",
                (chat_id, today, week_end)
            ) as cursor:
                reminders = await cursor.fetchall()
        return reminders

    async def get_reminders_by_date(self, chat_id, target_date):
        """Получение напоминаний на конкретную дату"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, reminder_text, event_date FROM reminders WHERE chat_id = ? AND event_date = ? ORDER BY event_date",
                (chat_id, target_date)
            ) as cursor:
                reminders = await cursor.fetchall()
        return reminders

    async def delete_reminder(self, reminder_id, chat_id):
        """Удаление напоминания"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM reminders WHERE id = ? AND chat_id = ?",
                (reminder_id, chat_id)
            )
            rows_affected = cursor.rowcount
            await db.commit()
        logger.info(f"Удалено напоминание {reminder_id} для chat_id {chat_id}, удалено строк: {rows_affected}")
        return rows_affected > 0

    async def update_reminder_text(self, reminder_id, chat_id, new_text):
        """Обновление текста напоминания"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE reminders SET reminder_text = ? WHERE id = ? AND chat_id = ?",
                (new_text, reminder_id, chat_id)
            )
            rows_affected = cursor.rowcount
            await db.commit()
        logger.info(f"Обновлен текст напоминания {reminder_id} для chat_id {chat_id}, изменено строк: {rows_affected}")
        return rows_affected > 0

    async def send_scheduled_reminders(self):
        """Фоновая задача для отправки напоминаний"""
        while True:
            try:
                now = datetime.now()
                current_time = now.time()
                current_date = now.date()
                
                # Вечерние напоминания в 18:00
                if current_time.hour == 18 and current_time.minute == 0:
                    tomorrow = current_date + timedelta(days=1)
                    await self._send_reminders_for_date(tomorrow, "вечер", "Завтра")
                
                # Утренние напоминания в 9:00  
                elif current_time.hour == 9 and current_time.minute == 0:
                    await self._send_reminders_for_date(current_date, "утро", "Сегодня")
                    
            except Exception as e:
                logger.error(f"Ошибка в фоновой задаче: {e}")
            
            await asyncio.sleep(60)

    async def _send_reminders_for_date(self, target_date, time_of_day, prefix):
        """Отправка напоминаний для конкретной даты"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT chat_id, reminder_text FROM reminders WHERE event_date = ?",
                (target_date,)
            ) as cursor:
                reminders = await cursor.fetchall()
                
            for chat_id, text in reminders:
                try:
                    message = f"🔔 {prefix}: {text}"
                    await self.bot.send_message(
                        chat_id=chat_id, 
                        text=message
                    )
                    logger.info(f"Отправлено {time_of_day}нее напоминание пользователю {chat_id}")
                except Exception as e:
                    logger.error(f"Ошибка отправки напоминания {chat_id}: {e}")
            
            # Удаляем прошедшие напоминания
            await db.execute("DELETE FROM reminders WHERE event_date < ?", (datetime.now().date(),))
            await db.commit()