import telebot
from dotenv import load_dotenv
from ai_operator import get_answer
import os
import sys
import time
import logging
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import datetime

logging.basicConfig(filename='log.txt', level=logging.INFO)

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

load_dotenv()
# 1. Инициализация бота
bot = telebot.TeleBot(os.getenv("TOKEN"))

# 2. Обработка сообщений
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "Здравствуйте, какой у вас вопрос?")

@bot.callback_query_handler(func=lambda call: str(call.from_user.id) == os.getenv("ADMIN_ID"))
def handle_admin_callbacks(call):
    if call.data == "log":
        with open("log.txt", "rb") as f:
            bot.send_document(call.message.chat.id, f)
    elif call.data == "clear_log":
        try:
            admin = call.from_user
            with open("log.txt", "w", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] logs was cleared by: @{admin.username} (ID: {admin.id})\n")

            bot.send_message(call.message.chat.id, "🧹 Логи успешно очищены.")
        except Exception as e:
            bot.send_message(call.message.chat.id, f"❌ Ошибка при очистке логов: {e}")
    elif call.data == "off":
        confirm_markup = InlineKeyboardMarkup()
        confirm_markup.add(
            InlineKeyboardButton("✅ Да", callback_data="confirm_off"),
            InlineKeyboardButton("❌ Нет", callback_data="cancel_off")
        )
        bot.send_message(call.message.chat.id, "Вы уверены, что хотите отключить бота?",
                         reply_markup=confirm_markup)
    elif call.data == "confirm_off":
        username = call.from_user.username or "unknown"
        admin_id = call.from_user.id

        try:
            with open("log.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.datetime.now()} - Bot was turned off by @{username} (ID: {admin_id})\n")

            bot.send_message(call.message.chat.id, "🔄 Выключение бота...")
            time.sleep(1)
            python = sys.executable
            os.execl(python, python, *sys.argv)

        except Exception as e:
            bot.send_message(call.message.chat.id, f"Error off: {e}")

    elif call.data == "cancel_off":
        bot.send_message(call.message.chat.id, "❌ Отключение отменено.")

@bot.message_handler(commands=['admin'])
def admin_panel(message):
    if str(message.from_user.id) != os.getenv("ADMIN_ID"):
        bot.send_message(message.chat.id, "⛔ У вас нет доступа.")
        return

    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("📥 Скачать лог", callback_data="log"))
    markup.add(InlineKeyboardButton("🧹 Очистить лог", callback_data="clear_log"))
    markup.add(InlineKeyboardButton("❌ Выключить бота", callback_data="off"))
    bot.send_message(message.chat.id, "🛠 Админ-панель", reply_markup=markup)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    response, score, isFound = get_answer(user_input)
    if isFound:
        bot.send_message(message.chat.id, response)
    else:
        bot.send_message(message.chat.id, response + os.getenv("OPERATOR"))
        user = message.from_user
        log_entry = (f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: @{user.username} | ID: {user.id} | is_bot: "
                     f"{user.is_bot} | Message: {user_input} | score: {score}")
        logging.info(log_entry)

# 3. Запуск
bot.polling()