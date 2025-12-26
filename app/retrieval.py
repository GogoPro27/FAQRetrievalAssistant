import json
import numpy as np
from pathlib import Path
from app.embeddings import create_embedding
from app.config import CONFIDENCE_THRESHOLD, TOP_K


def get_data_dir() -> Path:
    base_dir = Path(__file__).parent.parent
    return base_dir / "data"


def load_questions(data_dir: Path) -> list:
    with open(data_dir / "faqs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_answers(data_dir: Path) -> dict:
    with open(data_dir / "answers.json", "r", encoding="utf-8") as f:
        answers_list = json.load(f)
        return {a["answer_id"]: a["answer_text"] for a in answers_list}


def load_embeddings(data_dir: Path) -> np.ndarray:
    embeddings = np.load(data_dir / "embeddings.npy")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


DATA_DIR = get_data_dir()
QUESTIONS = load_questions(DATA_DIR)
ANSWERS = load_answers(DATA_DIR)
FAQ_EMBEDDINGS = load_embeddings(DATA_DIR)


def build_answer_to_en_question_map(questions) -> dict:
    mapping = {}
    for q in questions:
        if q.get("language") == "en":
            mapping[q["answer_id"]] = q

    if not mapping:
        raise ValueError("No English questions found in FAQ data")

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


def rank_questions(query: str) -> tuple[np.ndarray, np.ndarray]:
    if not query.strip():
        raise ValueError("Query must not be empty")

    query_embedding = create_embedding(query)

    if query_embedding.shape[0] != FAQ_EMBEDDINGS.shape[1]:
        raise ValueError("Embedding dimension mismatch")

    similarities = cosine_similarity(query_embedding)
    sorted_indices = np.argsort(similarities)[::-1]

    return sorted_indices, similarities


def collect_unique_answers(sorted_indices, similarities, top_k: int) -> tuple[list, np.ndarray]:
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
        if not selected_question:
            continue

        results.append({
            "id": selected_question["question_id"],
            "question": selected_question["question"],
            "answer": ANSWERS[answer_id],
            "similarity": float(similarities[idx]),
        })

        if len(results) == top_k:
            break

    return results, np.array(collected_similarities)


def search(query: str, top_k: int = TOP_K) -> dict:
    sorted_indices, similarities = rank_questions(query)

    results, collected_similarities = collect_unique_answers(
        sorted_indices,
        similarities,
        top_k
    )

    if not results:
        return {
            "results": [],
            "confidence": 0.0,
            "below_threshold": True
        }

    confidence = compute_confidence(collected_similarities)
    below_threshold = confidence < CONFIDENCE_THRESHOLD

    return {
        "results": results,
        "confidence": confidence,
        "below_threshold": below_threshold
    }
