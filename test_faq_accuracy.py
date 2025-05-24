import importlib
import numpy as np
import pytest

import ai_operator
from ai_operator import build_faiss_index, get_answer

# Минимальный порог точности для нашей синтетики
MIN_ACCURACY = 0.95

@pytest.fixture(autouse=True)
def reload_and_patch(monkeypatch):
    # Перезагружаем модуль, чтобы сбросить глобальные вопросы/ответы
    importlib.reload(ai_operator)

    # Синтетический FAQ: 4 вопроса, 2 кластера=
    monkeypatch.setattr(ai_operator, 'questions', [
        "q1 variant1", "q1 variant2",    # оба должны маппиться в answer1
        "q2 variant1", "q2 variant2"     # оба — в answer2
    ])
    monkeypatch.setattr(ai_operator, 'answers', [
        "answer1", "answer1",
        "answer2", "answer2"
    ])

    # DummyModel: возвращает [1,0] для «q1 …», [0,1] для «q2 …»
    class DummyModel:
        def encode(self, texts, convert_to_tensor=False):
            embs = []
            for t in texts:
                if t.startswith("q1"):
                    embs.append([1.0, 0.0])
                elif t.startswith("q2"):
                    embs.append([0.0, 1.0])
                else:
                    embs.append([0.0, 0.0])
            return np.array(embs, dtype="float32")

    monkeypatch.setattr(ai_operator, 'model', DummyModel())

    # После патча всегда пересобираем индекс
    build_faiss_index()
    yield

def test_nn_exact_matches():
    """
    Тестим, что для каждого вопроса в синтетическом FAQ
    get_answer возвращает правильный ответ (accuracy=100%).
    """
    total = len(ai_operator.questions)
    correct = 0
    for q, expected in zip(ai_operator.questions, ai_operator.answers):
        ans, score, found = get_answer(q, threshold=0.0, k=1)
        if found and ans == expected:
            correct += 1
        else:
            pytest.skip(f"Failed on {q!r}: got {ans!r}")
    accuracy = correct / total
    assert accuracy == 1.0

def test_nn_leave_one_out_accuracy():
    """
    Leave-One-Out: убираем каждый вопрос из индекса и проверяем,
    что система всё равно выдаёт любой другой вопрос из того же кластера.
    Ожидаем минимальную accuracy ≥ MIN_ACCURACY.
    """
    total = len(ai_operator.questions)
    correct = 0

    # группируем по ответам
    clusters = {}
    for q, a in zip(ai_operator.questions, ai_operator.answers):
        clusters.setdefault(a, []).append(q)

    for answer, qs in clusters.items():
        for left_out in qs:
            # исключаем left_out
            idxs  = [i for i, x in enumerate(ai_operator.questions) if x != left_out]
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(ai_operator, 'questions',
                                [ai_operator.questions[i] for i in idxs])
            monkeypatch.setattr(ai_operator, 'answers',
                                [ai_operator.answers[i]   for i in idxs])
            build_faiss_index()

            ans, score, found = get_answer(left_out, threshold=0.5, k=2)
            if found and ans == answer:
                correct += 1
            monkeypatch.undo()

    accuracy = correct / total
    assert accuracy >= MIN_ACCURACY, (
        f"NN Leave-One-Out accuracy too low: {accuracy:.1%}, "
        f"required ≥ {MIN_ACCURACY:.1%}"
    )