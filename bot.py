import telebot
from ai_operator import get_answer
import os
import sys
import time
import logging
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import datetime
from telebot.types import BotCommand
import json
from collections import defaultdict, deque

ENV_FILE = 'env.json'
def load_env():
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_env(data: dict):
   with open(ENV_FILE, 'w', encoding='utf-8') as f:
       json.dump(data, f, ensure_ascii=False, indent=2)

# загружаем переменные
env = load_env()
TOKEN = env['TOKEN']
OPERATOR = env['OPERATOR']
# ADMIN_ID="1264725550"
ADMINS_FILE = 'admins.json'

def load_admins():
    try:
        with open(ADMINS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return set(data.get('admins', []))
    except FileNotFoundError:
        return set()

def save_admins(admins_set):
    with open(ADMINS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"admins": list(admins_set)}, f, ensure_ascii=False, indent=2)

# Загрузка админов при старте
ADMIN_IDS = load_admins()

# конфиг файлов
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

last_user_message = {}

def is_duplicate(user_id: int, text: str) -> bool:
    """
    Проверяет, совпадает ли текущее сообщение (команда или текст) с последним от этого пользователя.
    Если да — возвращает True, иначе обновляет и возвращает False.
    """
    last = last_user_message.get(user_id)
    if last is not None and last == text:
        return True
    last_user_message[user_id] = text
    return False

# 1. Инициализация бота
bot = telebot.TeleBot(TOKEN)

# 2. Обработка сообщений
@bot.message_handler(commands=['start'])
def handle_start(message):
    user_id = message.from_user.id
    if is_duplicate(user_id, message.text):
        bot.send_message(
            message.chat.id,
            "⚠️ Пожалуйста, не отправляйте одну и ту же команду подряд."
        )
        return

    # Отдельный список команд для админа
    if str(user_id) in ADMIN_IDS:
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
    if str(call.from_user.id) in ADMIN_IDS:
        admin_logs = ['log','clear_log','off','confirm_off','cancel_off','add_faq','delete_faq', 'add_admin', 'remove_admin', 'list_admins', 'change_operator']
        if str(call.data) in admin_logs:
            handle_admin_callbacks(call)  # Вызываем вашу функцию для админа
            return

    if call.data == "help":
        bot.send_message(call.message.chat.id,
                         "Этот бот отвечает на распространенные вопросы по поселению "
                         "или связывает с оператором, если вопрос требует такого.\n"
                         "Не прикрепляйте файлы, картинки или гифки, иначе бот не ответит.\n"
                         "Не нужно описывать вашу ситацию - задавайте сразу конкретный вопрос.\n"
                         "Пример:\n"
                         "❌У меня есть долг по проживанию, как оплатить долг?\n"
                         "✅Как оплатить проживание?\n"
                         "Ниже прилагаю файл со всеми вопросами и ответами")
        create_help_list()
        with open("help_list.txt", "rb") as f:
            bot.send_document(call.message.chat.id, f)

    elif call.data == "faq":
        bot.send_message(call.message.chat.id,
        '"Как оформить переселение?":\n'
        'Для заполнения заявления на переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства\n'
        '"Как оплатить проживание?":\n'
        'Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru\n'
        '"Сколько стоит проживание?":\n'
        'Стоимость проживания в разных общежитиях разная и может со временем не значительно меняться, обычно от 1000 до 3000 руб./мес.\n')

    elif call.data == "feedback":
        # Запрос фидбэка с кнопкой «Назад»
        fb_markup = InlineKeyboardMarkup()
        fb_markup.add(InlineKeyboardButton("⏪ Назад", callback_data="cancel_feedback"))
        msg = bot.send_message(call.message.chat.id,
                                        "✏️ Напишите рекомендации по улучшению бота:",reply_markup = fb_markup)
        if call.data == "cancel_feedback":
            return
        bot.register_next_step_handler(msg, save_feedback)

    elif call.data == "cancel_feedback":
        # сбрасываем ожидающийся step-handler, чтобы ничего не сохранялось
        bot.clear_step_handler_by_chat_id(call.message.chat.id)
        # перерисовываем главное меню «help/faq/feedback/rating»
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("🛠 Помощь", callback_data="help"))
        markup.add(InlineKeyboardButton("❓ Часто задаваемые вопросы", callback_data="faq"))
        markup.add(InlineKeyboardButton("🚀 Обратная связь", callback_data="feedback"))
        markup.add(InlineKeyboardButton("🔥 Оценка работы бота", callback_data="rating"))
        bot.send_message(call.message.chat.id, "Вы вернулись в главное меню:", reply_markup=markup)

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

    else:
        bot.send_message(call.message.chat.id, "Вы более не являетесь админом или был вызван неизвестный(устаревший) колбек")
def save_feedback(message):
    user = message.from_user
    feedback_text = message.text
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(
            f"Feedback | Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: @{user.username} | ID: {user.id} | Feedback: {feedback_text}\n"
        )
    bot.send_message(message.chat.id, "✅ Спасибо за ваш отзыв!")

@bot.callback_query_handler(func=lambda call: str(call.from_user.id) in ADMIN_IDS)
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
        try:
            msg = bot.send_message(call.message.chat.id, "Введите новый вопрос:")
            bot.register_next_step_handler(msg, process_new_question)
        except Exception as e:
            bot.send_message(call.message.chat.id, f"Error off: {e}")
    # Удаление вопроса
    elif call.data == "delete_faq":
        msg = bot.send_message(call.message.chat.id, "Введите вопрос из help_list, который хотите удалить в таком формате: \"Как переселиться?\"")
        create_help_list()
        with open("help_list.txt", "rb") as f:
            bot.send_document(call.message.chat.id, f)
        bot.register_next_step_handler(msg, delete_faq)

    elif call.data == "add_admin":
        msg = bot.send_message(call.message.chat.id,
                               "Введите Telegram ID пользователя, которого вы хотите сделать админом:")
        bot.register_next_step_handler(msg, process_add_admin)

    elif call.data == "remove_admin":
        msg = bot.send_message(call.message.chat.id,
                               "Введите Telegram ID админа, которого нужно удалить:")
        bot.register_next_step_handler(msg, process_remove_admin)

    elif call.data == "list_admins":
        admins_list = "\n".join(f"• {aid}" for aid in sorted(ADMIN_IDS))
        bot.send_message(call.message.chat.id,
                                f"👥 Текущие администраторы:\n{admins_list}")

    elif call.data == "change_operator":
        msg = bot.send_message(call.message.chat.id,
                               "Введите новый Telegram-юзернейм оператора (с @):")
        bot.register_next_step_handler(msg, process_change_operator)

def process_change_operator(message):
    new_op = message.text.strip()
    if not new_op.startswith('@'):
        bot.send_message(message.chat.id, "❌ Юзернейм должен начинаться с @. Попробуйте ещё раз.")
        return

    # обновляем env.json
    env['OPERATOR'] = new_op
    save_env(env)

    # обновляем переменную в памяти
    global OPERATOR
    OPERATOR = new_op

    bot.send_message(message.chat.id,
                     f"✅ Оператор успешно изменён на {new_op}.")

def process_add_admin(message):
    new_admin_id = message.text.strip()
    if not new_admin_id.isdigit():
        bot.send_message(message.chat.id, "❌ Неверный формат ID. Попробуйте ещё раз.")
        return
    if new_admin_id in ADMIN_IDS:
        bot.send_message(message.chat.id, "ℹ️ Этот пользователь уже является админом.")
        return

    ADMIN_IDS.add(new_admin_id)
    save_admins(ADMIN_IDS)
    bot.send_message(message.chat.id,
                     f"✅ Пользователь с ID `{new_admin_id}` добавлен в список админов.")

def process_remove_admin(message):
    remove_id = message.text.strip()
    if not remove_id.isdigit():
        bot.send_message(message.chat.id, "❌ Неверный формат ID. Попробуйте ещё раз.")
        return

    if remove_id not in ADMIN_IDS:
        bot.send_message(message.chat.id, "ℹ️ Пользователь с таким ID не является админом.")
        return

    if remove_id == str(message.from_user.id):
        bot.send_message(message.chat.id, "❌ Вы не можете удалить сами себя.")
        return

    if remove_id == "1264725550":
        bot.send_message(message.chat.id, "❌ Вы не можете удалить этого админа.")
        return

    ADMIN_IDS.remove(remove_id)
    save_admins(ADMIN_IDS)
    bot.send_message(message.chat.id,
                     f"✅ Пользователь с ID `{remove_id}` удалён из списка админов.")

def create_help_list():
    # Чтение из файла и парсинг
    with open('faq_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    header = "Формат списка - Вопрос: Ответ\n\n"

    with open('help_list.txt', 'w', encoding='utf-8') as help_out:
        help_out.write(header)

        for pair in enumerate(data.items(), start=1):
            item_string = str(pair[0]) + ". " + ": ".join(pair[1]) + "\n\n"
            help_out.write(item_string)

#Добавление и удаление вопросов в faq_data
def process_new_question(message):
    new_question = message.text
    if not new_question:
        bot.send_message(message.chat.id, "❌ Вопрос не может быть пустым. Не добавляйте gif/png/стикеры или любой другой не текстовый формат. Добавление отменено.")
        return
    msg = bot.send_message(message.chat.id, "Введите ответ на этот вопрос:")
    bot.register_next_step_handler(msg, lambda m: save_new_faq(new_question, m))

def save_new_faq(question, message):
    from ai_operator import faq_data, save_faq_data, questions, answers, build_faiss_index

    answer = message.text
    if not answer:
        bot.send_message(message.chat.id, "❌ Ответ не может быть пустым. Не добавляйте gif/png/стикеры или любой другой не текстовый формат. Добавление отменено.")
        return
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
    user_id = message.from_user.id
    if is_duplicate(user_id, message.text):
        bot.send_message(
            message.chat.id,
            "⚠️ Пожалуйста, не отправляйте одну и ту же команду подряд."
        )
        return
    if str(user_id) not in ADMIN_IDS:
        bot.send_message(message.chat.id, "⛔ У вас нет доступа.")
        return

    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("📥 Скачать лог", callback_data="log"))
    markup.add(InlineKeyboardButton("🧹 Очистить лог", callback_data="clear_log"))
    markup.add(InlineKeyboardButton("❌ Выключить бота", callback_data="off"))
    markup.add(InlineKeyboardButton("➕ Добавить вопрос", callback_data="add_faq"))
    markup.add(InlineKeyboardButton("➖ Удалить вопрос", callback_data="delete_faq"))
    markup.add(InlineKeyboardButton("🔑 Добавить админа", callback_data="add_admin"))
    markup.add(InlineKeyboardButton("🔓 Удалить админа", callback_data="remove_admin"))
    markup.add(InlineKeyboardButton("👥 Список админов", callback_data="list_admins"))
    markup.add(InlineKeyboardButton("✏️ Изменить оператора", callback_data="change_operator"))
    bot.send_message(message.chat.id, "🛠 Админ-панель", reply_markup=markup)

#Основной перехватчик сообщений и его вспомогательные функции
RATE_LIMIT_COUNT = 4
RATE_LIMIT_WINDOW = 10.0

user_messages = defaultdict(lambda: deque())

def is_allowed(user_id: int) -> bool:
    now = time.time()
    dq = user_messages[user_id]
    # Удаляем старые отметки
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_COUNT:
        return False
    dq.append(now)
    return True

MAX_MESSAGE_LENGTH = 400
MIN_MESSAGE_LENGTH = 4

def validate_length(text: str, chat_id: int) -> bool:
    length = len(text)
    if length > MAX_MESSAGE_LENGTH:
        bot.send_message(
            chat_id,
            f"❗️ Сообщение слишком длинное ({length} символов).\n"
            f"Максимум — {MAX_MESSAGE_LENGTH} символов. Пожалуйста, сократите ваш текст."
        )
        return False
    if length < MIN_MESSAGE_LENGTH:
        bot.send_message(
            chat_id,
            f"❗️ Сообщение слишком короткое ({length} символов).\n"
            f"Минимум — {MIN_MESSAGE_LENGTH} символов."
        )
        return False
    return True

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    uid = message.from_user.id

    # Проверка на дубликат (команды обработаны выше)
    if is_duplicate(uid, user_input):
        bot.send_message(
            message.chat.id,
            "⚠️ Пожалуйста, не отправляйте два одинаковых сообщения подряд."
        )
        return

    if not validate_length(user_input, message.chat.id):
        return

    uid = message.from_user.id
    if not is_allowed(uid):
        bot.send_message(
            message.chat.id,
            "⚠️ Слишком часто отправляете сообщения — подождите пару секунд."
        )
        return

    response, score, isFound, parseMode = get_answer(user_input)
    if response is None:
        response = "Ошибка: message text is empty"
    if isFound:
        bot.send_message(message.chat.id, response, parse_mode=parseMode)
    else:
        bot.send_message(message.chat.id, response + OPERATOR, parse_mode=parseMode)
        user = message.from_user

        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(
            f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: @{user.username} | ID: {user.id} | "
            f"Message: {user_input} | score: {score}\n")

# 3. Запуск
if __name__ == "__main__":
    bot.polling()