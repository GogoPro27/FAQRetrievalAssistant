import os
import numpy as np
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from dotenv import load_dotenv
from app.config import EMBEDDING_MODEL

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


def validate_text(text: str) -> None:
    """Validate that text is not empty or whitespace-only."""
    if not text or not text.strip():
        raise ValueError("Cannot create embedding for empty text")


def call_embedding_api(text: str) -> CreateEmbeddingResponse:
    """Call OpenAI embeddings API and handle errors."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
    except Exception as e:
        raise RuntimeError("Embedding generation failed") from e
    return response


def create_embedding(text: str, normalize: bool = True) -> np.ndarray:
    """Generate and optionally normalize embedding vector for text."""
    validate_text(text)

    response = call_embedding_api(text)

    embedding = np.array(response.data[0].embedding, dtype=np.float32)

    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    return embedding
