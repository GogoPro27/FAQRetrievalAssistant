import json
import numpy as np
import os
from app.embeddings import create_embedding

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FAQS_PATH = os.path.join(DATA_DIR, "faqs.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

def main():
    with open(FAQS_PATH, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    embeddings = []
    for i, faq in enumerate(faqs, 1):
        embeddings.append(create_embedding(faq["question"]))
        print(f"   Processed {i}/{len(faqs)} FAQs")

    np.save(EMBEDDINGS_PATH, np.vstack(embeddings))
    print(f"Saved in: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
