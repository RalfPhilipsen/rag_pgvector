from openai import OpenAI

from .config import settings

_client = OpenAI(
    base_url=settings.lmstudio_base_url,
    api_key=settings.lmstudio_api_key,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the answer is not contained in the context, say you "
    "don't know."
)


def answer_with_context(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    resp = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content or ""
