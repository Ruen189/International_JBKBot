import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

# 1. База вопросов и ответов
FAQ_FILE = "faq_data.json"
def load_faq_data():
    if not os.path.exists(FAQ_FILE):
        return {}
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_faq_data(faq_data):
    with open(FAQ_FILE, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)

# Загружаем базу
faq_data = load_faq_data()
questions = list(faq_data.keys())
answers = list(faq_data.values())

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
def get_answer(query, threshold=0.9, k=1):
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