from openai import OpenAI

from .config import settings

_client = OpenAI(
    base_url=settings.lmstudio_base_url,
    api_key=settings.lmstudio_api_key,
)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _client.embeddings.create(model=settings.embedding_model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out
