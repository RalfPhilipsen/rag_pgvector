from enum import Enum

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import embeddings as lm_embeddings
from .config import settings


class Chunker(str, Enum):
    recursive = "recursive"
    semantic = "semantic"


class _LMStudioEmbeddings(Embeddings):
    """LangChain Embeddings adapter that delegates to the LM Studio client."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return lm_embeddings.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return lm_embeddings.embed_texts([text])[0]


_recursive = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

_semantic = SemanticChunker(
    embeddings=_LMStudioEmbeddings(),
    breakpoint_threshold_type="percentile",
)


def split(text: str, chunker: Chunker) -> list[str]:
    if chunker is Chunker.recursive:
        return _recursive.split_text(text)
    if chunker is Chunker.semantic:
        return _semantic.split_text(text)
    raise ValueError(f"Unknown chunker: {chunker}")
