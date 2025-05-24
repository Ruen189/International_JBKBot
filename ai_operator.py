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

# Инициализация эмбеддингов
question_embeddings = None
index = None

def build_faiss_index():
    global question_embeddings, index

    # Пересоздаем эмбеддинги и индекс
    question_embeddings = model.encode(questions, convert_to_tensor=False)
    question_embeddings = np.array(question_embeddings).astype("float32")

    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(question_embeddings)

build_faiss_index()

import numpy as np

def get_answer(query, threshold=0.85, k=10):
    """
    Ищет ответ для query:
      - k         — число кандидатов из FAISS по L2;
      - threshold — минимальная косинусная схожесть для отдачи ответа.
    """
    global index, question_embeddings, answers, model

    # 1. Эмбеддинг запроса
    query_embedding = model.encode([query]).astype("float32")

    # 2. Берём k ближайших по L2
    distances, indices = index.search(query_embedding, k)
    candidate_idxs = indices[0]

    # 3. Нормируем эмбеддинги для косинуса
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    questions_norm = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

    # 4. Считаем косинусы только для кандидатов
    cos_all = np.dot(questions_norm, query_norm.T).squeeze()
    candidate_sims = cos_all[candidate_idxs]

    # 5. Выбираем лучший среди кандидатов
    best_local_pos = int(np.argmax(candidate_sims))
    best_idx       = int(candidate_idxs[best_local_pos])
    best_score     = candidate_sims[best_local_pos]

    # 6. Фильтрация по порогу
    if best_score >= threshold:
        return answers[best_idx], best_score, True
    else:
        return "Извините, я не смог найти подходящего ответа. Напишите оператору:", best_score, False