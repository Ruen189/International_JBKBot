import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
# 1. Инициализация бота
bot = telebot.TeleBot(os.getenv("TOKEN"))

# 2. Шаблонные вопросы и ответы
faq_data = {
    "Как заселиться в общежитие?": "Для заселения вам нужно обратиться в студгородок с паспортом и студенческим билетом",
    "Какие документы нужны для заселения?": "Паспорт, студенческий билет, договор на обучение и медицинская справка.",
    "Сколько стоит проживание?": "Стоимость проживания зависит от типа общежития, обычно от 1000 до 3000 руб./мес.",
    "Можно ли выбрать комнату?": "Выбор комнаты ограничен, уточните в деканате или студгородке.",
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# 3. NLP-модель (TF-IDF)
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)


# 4. Обработка сообщений
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "Здравствуйте, какой у вас вопрос?")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarities.argmax()
    confidence = similarities[0, best_match_index]

    if confidence > 0.5:
        response = answers[best_match_index]
    else:
        response = "Извините, я не понял ваш вопрос. Вы можете его уточнить у https://t.me/Yaroslav9605"

    bot.send_message(message.chat.id, response)


# 5. Запуск
bot.polling()
