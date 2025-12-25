import json
import os
import numpy as np
from app.embeddings import create_embedding

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FAQS_PATH = os.path.join(DATA_DIR, "faqs.json")
ANSWERS_PATH = os.path.join(DATA_DIR, "answers.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

CONFIDENCE_THRESHOLD = 0.7

with open(FAQS_PATH, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
    ANSWERS = {a["answer_id"]: a["answer_text"] for a in json.load(f)}

FAQ_EMBEDDINGS = np.load(EMBEDDINGS_PATH)
FAQ_EMBEDDINGS = FAQ_EMBEDDINGS / np.linalg.norm(
    FAQ_EMBEDDINGS, axis=1, keepdims=True
)


def build_answer_to_en_question_map(questions):
    mapping = {}
    for q in questions:
        if q["language"] == "en":
            mapping[q["answer_id"]] = q
    return mapping


ANSWER_TO_EN_QUESTION = build_answer_to_en_question_map(QUESTIONS)


def cosine_similarity(query_embedding: np.ndarray) -> np.ndarray:
    return FAQ_EMBEDDINGS @ query_embedding


def compute_confidence(similarities: np.ndarray) -> float:
    if len(similarities) < 2:
        return 0.0

    s1 = float(similarities[0])
    s2 = float(similarities[1])
    margin = s1 - s2

    similarity_component = max(0.0, min(1.0, (s1 - 0.5) / 0.5))
    margin_component = max(0.0, min(1.0, margin / 0.15))
    confidence = 0.7 * similarity_component + 0.3 * margin_component

    return round(confidence, 3)


def rank_questions(query: str):
    query_embedding = create_embedding(query)
    similarities = cosine_similarity(query_embedding)
    sorted_indices = np.argsort(similarities)[::-1]

    return sorted_indices, similarities


def collect_unique_answers(sorted_indices, similarities, top_k: int):
    results = []
    seen_answers = set()
    collected_similarities = []

    for idx in sorted_indices:
        q = QUESTIONS[idx]
        answer_id = q["answer_id"]

        if answer_id in seen_answers:
            continue

        seen_answers.add(answer_id)
        collected_similarities.append(similarities[idx])

        selected_question = ANSWER_TO_EN_QUESTION[answer_id]

        results.append({
            "id": selected_question["question_id"],
            "question": selected_question["question"],
            "answer": ANSWERS[answer_id],
            "similarity": float(similarities[idx]),
        })

        if len(results) == top_k:
            break

    return results, np.array(collected_similarities)


def search(query: str, top_k: int = 3):
    sorted_indices, similarities = rank_questions(query)

    results, collected_similarities = collect_unique_answers(
        sorted_indices,
        similarities,
        top_k
    )

    confidence = compute_confidence(collected_similarities)
    below_threshold = confidence < CONFIDENCE_THRESHOLD

    return {
        "results": results,
        "confidence": confidence,
        "below_threshold": below_threshold
    }
