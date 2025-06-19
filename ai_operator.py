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
    question_embeddings = np.array(question_embeddings).astype("float16")

    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(question_embeddings)

build_faiss_index()

def get_answer(query, threshold=0.75, k=10, num_answers=3, parse_mode = None):
    """
    Ищет один или несколько ответов для query:
      - k           — число кандидатов из FAISS по L2;
      - threshold   — минимальная косинусная схожесть для отдачи ответа;
      - num_answers — максимальное число вариантов в списке.
    Правила:
      - Если лучший кандидат имеет score > 0.9, возвращаем только его.
      - Иначе, если лучший score >= threshold, возвращаем нумерованный список всех Q&A,
        у которых схожесть >= threshold.
      - В списке сначала вопрос в кавычках, ниже — ответ.
      - Повторяющиеся ответы убираются (только первый).
    """
    global index, question_embeddings, answers, questions, model

    # 1. Эмбеддинг запроса
    query_embedding = model.encode([query]).astype("float16")

    # 2. Берём k ближайших по L2
    distances, indices = index.search(query_embedding, k)
    print(query_embedding)
    candidate_idxs = indices[0]

    # 3. Нормализуем эмбеддинги для косинуса
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    questions_norm = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

    # 4. Считаем косинусы только для кандидатов
    cos_all = np.dot(questions_norm, query_norm.T).squeeze()
    # Составляем список (idx, sim) для каждого кандидата
    candidates = [(idx, float(cos_all[idx])) for idx in candidate_idxs]
    # Сортируем по убыванию схожести
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Самый лучший
    best_idx, best_score = candidates[0]
    # Если очень высокая схожесть — один ответ
    if best_score > 0.9:
        return answers[best_idx], best_score, True, parse_mode

    # Если хотя бы порог пройден — несколько вариантов
    if best_score >= threshold:
        filtered = [(idx, sim) for idx, sim in candidates if sim >= threshold]
        seen_answers = set()
        qa_list = []
        for idx, sim in filtered:
            ans = answers[idx]
            if ans in seen_answers:
                continue
            seen_answers.add(ans)
            q = questions[idx]
            qa_list.append((q, ans))
            if len(qa_list) >= num_answers:
                break

        # Формируем ответ в зависимости от числа вариантов
        if len(qa_list) == 1:
            q, a = qa_list[0]
            response = (
                f"<i>Возможно, вы имели в виду: «{q}»?</i>\n"
                f"<b>Ответ:</b>\n{a}"
            )
            return response, best_score, True, 'HTML'
        else:
            lines = []
            for i, (q, a) in enumerate(qa_list, start=1):
                lines.append(f"{i}. \"{q}\"\n{a}")
            response = "Возможно, вы имели в виду:\n\n" + "\n\n".join(lines)

        return response, best_score, True, parse_mode

    # Иначе — отказ
    return (
        "Извините, я не смог найти подходящего ответа.\n"
        "Попробуйте написать вопрос проще (в одно предложение),\n"
        "нажмите кнопку помощи или напишите оператору: ",
        best_score,
        False,
        parse_mode
    )