import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from app.embeddings import create_embedding

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FAQS_PATH = os.path.join(DATA_DIR, "faqs.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(FAQS_PATH, "r", encoding="utf-8") as file:
    FAQS = json.load(file)

FAQ_EMBEDDINGS = np.load(EMBEDDINGS_PATH)

FAQ_EMBEDDINGS = FAQ_EMBEDDINGS / np.linalg.norm(
    FAQ_EMBEDDINGS, axis=1, keepdims=True
)


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


def cosine_similarity(query_embedding: np.ndarray) -> np.ndarray:
    return FAQ_EMBEDDINGS @ query_embedding


def format_search_results(indices: np.ndarray, similarities: np.ndarray) -> list[dict]:
    results = []
    for idx in indices:
        faq = FAQS[idx]
        results.append({
            "id": faq["id"],
            "question": faq["question"],
            "answer": faq["answer"],
            "similarity": float(similarities[idx])
        })
    return results


def search(query: str, top_k: int = 3):
    query_embedding = create_embedding(query)
    similarities = cosine_similarity(query_embedding)

    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_k]

    results = format_search_results(top_indices, similarities)

    top_similarities = similarities[sorted_indices][:2]
    confidence = compute_confidence(top_similarities)

    return {
        "results": results,
        "confidence": confidence
    }
