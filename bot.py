import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from ai_operator import get_answer
import os

load_dotenv()
# 1. Инициализация бота
bot = telebot.TeleBot(os.getenv("TOKEN"))

# 2. Обработка сообщений
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "Здравствуйте, какой у вас вопрос?")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    response, score, isFound = get_answer(user_input)
    if isFound:
        bot.send_message(message.chat.id, response)
    else:
        bot.send_message(message.chat.id, response + os.getenv("OPERATOR"))

# 3. Запуск
bot.polling()