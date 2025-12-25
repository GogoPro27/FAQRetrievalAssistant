import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FAQS_PATH = "data/faqs.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_faqs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def main():
    faqs = load_faqs(FAQS_PATH)
    embeddings = []

    for i, faq in enumerate(faqs, 1):
        question = faq["question"]
        embedding = create_embedding(question)
        embeddings.append(embedding)
        print(f"   Processed {i}/{len(faqs)} FAQs")

    print()

    embeddings_matrix = np.vstack(embeddings)

    np.save(EMBEDDINGS_PATH, embeddings_matrix)

    print(f"Saved in: {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings_matrix.shape}")


if __name__ == "__main__":
    main()
