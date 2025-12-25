import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from app.embeddings import create_embedding

load_dotenv()

DATA_DIR = "data"
FAQS_PATH = os.path.join(DATA_DIR, "faqs.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(FAQS_PATH, "r", encoding="utf-8") as file:
    FAQS = json.load(file)

FAQ_EMBEDDINGS = np.load(EMBEDDINGS_PATH)

FAQ_EMBEDDINGS = FAQ_EMBEDDINGS / np.linalg.norm(
    FAQ_EMBEDDINGS, axis=1, keepdims=True
)


def cosine_similarity(query_embedding: np.ndarray) -> np.ndarray:
    return FAQ_EMBEDDINGS @ query_embedding


def search(query: str, top_k: int = 3):
    query_embedding = create_embedding(query)
    similarities = cosine_similarity(query_embedding)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        faq = FAQS[idx]
        results.append({
            "question": faq["question"],
            "answer": faq["answer"],
            "similarity": float(similarities[idx])
        })

    return results
