import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_embedding(text: str, normalize: bool = True) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    embedding = np.array(response.data[0].embedding, dtype=np.float32)

    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    return embedding
