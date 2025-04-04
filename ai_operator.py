import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. База вопросов и ответов в виде словаря.
faq_data = {
    # Группа 1: Переселение
    "Как переселиться?": "Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как оформить переселение?": "Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как переселиться в другое общежитие?": "Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как поменять комнату?": "Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как переселиться к другу?": "Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как поменять общежитие?": "Для заполнения заявления не переселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",

    # Группа 2: Поселение
    "Как поселиться в общежитие?": "Для заполнения заявления на поселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как поселиться?": "Для заполнения заявления на поселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как поселиться впервые?": "Для заполнения заявления на поселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как заселиться в общагу?": "Для заполнения заявления на поселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как оформить поселение?": "Для заполнения заявления на поселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",
    "Как получить общагу?": "Для заполнения заявления на поселение необходимо подойти в жилищно-бытовую комиссию во время дежурства",

    # Группа 3: Оплата
    "Как оплатить проживание?": "Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru",
    "Как внести оплату за общежитие?": "Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru",
    "Как оплатить общагу?": "Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru",
    "Как оплатить долг?": "Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru",
    "Как внести деньги за проживание?": "Необходимо зайти в ЛК студента в раздел \"Платежи и задолжности\" и оплатить через сервис pay.urfu.ru",

    # Группа 4: Этапы поселения
    "Что нужно сделать для поселения?": "С этапами поселения можно ознакомиться в памятке",
    "Какие этапы поселения?": "С этапами поселения можно ознакомиться в памятке",
    "Что делать после получения документов?": "С этапами поселения можно ознакомиться в памятке",
    "Как оформить проживание?": "С этапами поселения можно ознакомиться в памятке",

    # Группа 5: Конфликт с соседом
    "Что делать в случае конфликта с соседом?": "Для выяснения ситуации Вам необходимо подойти в жилищно-бытовую комиссию в дни дежурства или обратиться в ЛС к ...",
    "Куда обратиться, если появился неоправданный долг за общежитие?": "Для выяснения ситуации Вам необходимо подойти в жилищно-бытовую комиссию в дни дежурства или обратиться в ЛС к ...",
    "Меня поселили в полную комнату, что делать?": "Для выяснения ситуации Вам необходимо подойти в жилищно-бытовую комиссию в дни дежурства или обратиться в ЛС к ...",
    "Что делать, если у меня не начисляется долг за общежитие?": "Для выяснения ситуации Вам необходимо подойти в жилищно-бытовую комиссию в дни дежурства или обратиться в ЛС к ...",

    # Группа 6: Одноместное поселение в НВК
    "Как оформить одноместное поселение в НВК?": "Вам необходимо заполнить заявление на одноместное поселение (как-то прикрепить файл) и принести его в жилищно-бытовую комиссию в дни дежурства",
    "Как оформить одноместное?": "Вам необходимо заполнить заявление на одноместное поселение (как-то прикрепить файл) и принести его в жилищно-бытовую комиссию в дни дежурства",
    "Как жить одному в НВК?": "Вам необходимо заполнить заявление на одноместное поселение (как-то прикрепить файл) и принести его в жилищно-бытовую комиссию в дни дежурства",
    "Как поселиться одному в НВК?": "Вам необходимо заполнить заявление на одноместное поселение (как-то прикрепить файл) и принести его в жилищно-бытовую комиссию в дни дежурства",

    # Группа 7: Расписание комиссии
    "Когда работает комиссия?": "Комиссия работает в ... (вставить расписание)",
    "Когда дежурит ЖБК?": "Комиссия работает в ... (вставить расписание)",
    "Когда дежурит комиссия?": "Комиссия работает в ... (вставить расписание)",
    "Когда работает ЖБК?": "Комиссия работает в ... (вставить расписание)",
    "В какие дни работает ЖБК?": "Комиссия работает в ... (вставить расписание)",
    "Какие дни дежурства комиссии?": "Комиссия работает в ... (вставить расписание)",
    "Когда можно прийти в ЖБК?": "Комиссия работает в ... (вставить расписание)",
    "Где посмотреть расписание работы комиссии?": "Комиссия работает в ... (вставить расписание)",
    "Какое расписание работы комиссии?": "Комиссия работает в ... (вставить расписание)",

    # Группа 8: Местоположение ЖБК
    "Где находится ЖБК?": "Комиссия работает в ... (указать место)",
    "Где работает комиссия?": "Комиссия работает в ... (указать место)",
    "Где дежурит ЖБК?": "Комиссия работает в ... (указать место)",
    "Где дежурит комиссия?": "Комиссия работает в ... (указать место)",
    "В каком кабинете находится ЖБК?": "Комиссия работает в ... (указать место)",
}

# 2. Извлекаем списки вопросов и ответов из словаря.
questions = list(faq_data.keys())
answers = list(faq_data.values())

# 3. Загружаем модель для получения эмбеддингов.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Вычисляем эмбеддинги для всех вопросов.
question_embeddings = model.encode(questions, convert_to_tensor=False)
question_embeddings = np.array(question_embeddings).astype("float32")

# Определяем размерность эмбеддингов.
dimension = question_embeddings.shape[1]

# 4. Создаем индекс Faiss для быстрого поиска.
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)


# 5. Функция для поиска ответа по входящему вопросу.
def get_answer(query, threshold=0.7, k=1):
    # Вычисляем эмбеддинг для запроса.
    isFound = False
    query_embedding = model.encode([query]).astype("float32")

    # Поиск ближайших соседей.
    distances, indices = index.search(query_embedding, k)

    # Дополнительное вычисление косинусного сходства.
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    questions_norm = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)
    cos_similarities = np.dot(questions_norm, query_norm.T).squeeze()

    # Определяем индекс наиболее похожего вопроса.
    best_idx = int(np.argmax(cos_similarities))
    best_score = cos_similarities[best_idx]

    if best_score >= threshold:
        isFound = True
        return answers[best_idx], best_score, isFound
    else:
        return "Извините, я не смог найти подходящего ответа. Напишите оператору:", best_score, isFound