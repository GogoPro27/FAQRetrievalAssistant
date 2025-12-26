import json
import numpy as np
from pathlib import Path
from app.embeddings import create_embedding


def get_data_dir() -> Path:
    """Return the data directory path."""
    base_dir = Path(__file__).parent.parent
    return base_dir / "data"


DATA_DIR = get_data_dir()
FAQS_PATH = DATA_DIR / "faqs.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"


def load_faqs() -> list:
    """Load FAQ questions from JSON file."""
    with open(FAQS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_faq(faq, index) -> None:
    """Validate that FAQ entry has non-empty question text."""
    if not faq["question"].strip():
        raise ValueError(f"Invalid FAQ entry at index {index}")


def validate_embeddings(embeddings, faqs) -> None:
    """Verify that embeddings were generated for all FAQs."""
    if embeddings.shape[0] != len(faqs):
        raise RuntimeError("Embedding generation incomplete")


def main() -> None:
    """Generate embeddings for all FAQ questions and save to disk."""
    faqs = load_faqs()

    embeddings = []
    for i, faq in enumerate(faqs, 1):
        validate_faq(faq, i)
        embeddings.append(create_embedding(faq["question"]))
        print(f"Processed {i}/{len(faqs)} FAQs")

    validate_embeddings(np.vstack(embeddings), faqs)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved in: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
