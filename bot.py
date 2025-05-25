import telebot
from dotenv import load_dotenv
from ai_operator import get_answer
import os
import sys
import time
import logging
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import datetime
from telebot.types import BotCommand

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
    user_id = message.from_user.id
    # Отдельный список команд для админа
    if str(user_id) == os.getenv("ADMIN_ID"):
        bot.set_my_commands([
            BotCommand("start", "Начать работу"),
            BotCommand("admin", "Открыть админ панель"),
        ], scope=telebot.types.BotCommandScopeChat(chat_id=user_id))
    else:
        bot.set_my_commands([
            BotCommand("start", "Начать работу"),
        ], scope=telebot.types.BotCommandScopeChat(chat_id=user_id))
    # Создаем меню кнопок
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("🛠 Помощь", callback_data="help"))
    markup.add(InlineKeyboardButton("❓ Часто задаваемые вопросы", callback_data="faq"))
    markup.add(InlineKeyboardButton("🚀 Обратная связь", callback_data="feedback"))
    markup.add(InlineKeyboardButton("🔥 Оценка работы бота", callback_data="rating"))
    # Отправляем приветствие с кнопками
    bot.send_message(message.chat.id, "Здравствуйте, какой у вас вопрос?", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def handle_all_callbacks(call):
    if str(call.from_user.id) == os.getenv("ADMIN_ID"):
        admin_logs = ['log','clear_log','off','confirm_off','cancel_off','add_faq','delete_faq']
        if str(call.data) in admin_logs:
            handle_admin_callbacks(call)  # Вызываем вашу функцию для админа
            return

    if call.data == "help":
        bot.send_message(call.message.chat.id,
                         "Этот бот отвечает на распространенные вопросы по поселению "
                         "или связывает с оператором, если вопрос требует такого.\n"
                         "Не нужно описывать ваше текущее состоянии или ситуацию - задавайте сразу конкретный вопрос.\n"
                         "Пример:\n"
                         "❌У меня есть долг по проживанию, как оплатить долг?\n"
                         "✅Как оплатить проживание?")

    elif call.data == "faq":
        bot.send_message(call.message.chat.id,
        '"Как оформить переселение?":\n'
        'Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства\n'
        '"Как оплатить проживание?":\n'
        'Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru\n'
        '"Можно ли выбрать комнату?":\n'
        'Выбор комнаты ограничен, уточните в деканате или студгородке.\n')

    elif call.data == "feedback":
        msg = bot.send_message(call.message.chat.id, "✏️ Напишите рекомендации по улучшению бота:")
        bot.register_next_step_handler(msg, save_feedback)

    elif call.data == "rating":
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("⭐", callback_data="rate_1")),
        markup.add(InlineKeyboardButton("⭐⭐", callback_data="rate_2")),
        markup.add(InlineKeyboardButton("⭐⭐⭐", callback_data="rate_3")),
        markup.add(InlineKeyboardButton("⭐⭐⭐⭐", callback_data="rate_4")),
        markup.add(InlineKeyboardButton("⭐⭐⭐⭐⭐", callback_data="rate_5")),
        bot.send_message(call.message.chat.id, "Пожалуйста, оцените работу бота:", reply_markup=markup)

    elif call.data.startswith("rate_"):
        rating_value = call.data.split("_")[1]  # Получаем число из callback_data
        user = call.from_user
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(
                f"Rating | Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: @{user.username} | ID: {user.id} | Rating: {rating_value}\n"
            )
        bot.send_message(call.message.chat.id, f"✅ Спасибо за вашу оценку: {rating_value} ⭐!")

def save_feedback(message):
    user = message.from_user
    feedback_text = message.text
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(
            f"Feedback | Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: @{user.username} | ID: {user.id} | Feedback: {feedback_text}\n"
        )
    bot.send_message(message.chat.id, "✅ Спасибо за ваш отзыв!")

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
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Логи были очищены пользователем: @{admin.username} (ID: {admin.id})\n")

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
    # Добавление нового вопроса
    elif call.data == "add_faq":
        msg = bot.send_message(call.message.chat.id, "Введите новый вопрос:")
        bot.register_next_step_handler(msg, process_new_question)

    # Удаление вопроса
    elif call.data == "delete_faq":
        msg = bot.send_message(call.message.chat.id, "Введите точный текст вопроса, который хотите удалить:")
        bot.register_next_step_handler(msg, delete_faq)

#Добавление и удаление вопросов в faq_data
def process_new_question(message):
    new_question = message.text
    msg = bot.send_message(message.chat.id, "Введите ответ на этот вопрос:")
    bot.register_next_step_handler(msg, lambda m: save_new_faq(new_question, m))

def save_new_faq(question, message):
    from ai_operator import faq_data, save_faq_data, questions, answers, build_faiss_index

    answer = message.text
    faq_data[question] = answer
    save_faq_data(faq_data)

    questions.clear()
    questions.extend(faq_data.keys())
    answers.clear()
    answers.extend(faq_data.values())

    build_faiss_index()

    bot.send_message(message.chat.id, f"✅ Добавлен новый вопрос:\n\n{question}\nОтвет:\n{answer}")


def delete_faq(message):
    from ai_operator import faq_data, save_faq_data, questions, answers, build_faiss_index

    question = message.text
    if question in faq_data:
        del faq_data[question]
        save_faq_data(faq_data)

        questions.clear()
        questions.extend(faq_data.keys())
        answers.clear()
        answers.extend(faq_data.values())

        build_faiss_index()

        bot.send_message(message.chat.id, f"✅ Вопрос удалён:\n{question}")
    else:
        bot.send_message(message.chat.id, "❌ Такого вопроса нет в базе.")


#Админ панель
@bot.message_handler(commands=['admin'])
def admin_panel(message):
    if str(message.from_user.id) != os.getenv("ADMIN_ID"):
        bot.send_message(message.chat.id, "⛔ У вас нет доступа.")
        return

    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("📥 Скачать лог", callback_data="log"))
    markup.add(InlineKeyboardButton("🧹 Очистить лог", callback_data="clear_log"))
    markup.add(InlineKeyboardButton("❌ Выключить бота", callback_data="off"))
    markup.add(InlineKeyboardButton("➕ Добавить вопрос", callback_data="add_faq"))
    markup.add(InlineKeyboardButton("➖ Удалить вопрос", callback_data="delete_faq"))
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

        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(
            f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: @{user.username} | ID: {user.id} | "
            f"Message: {user_input} | score: {score}\n")

# 3. Запуск
if __name__ == "__main__":
    bot.remove_webhook()
    time.sleep(1)
    bot.infinity_polling()