from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import chunking, db, embeddings, llm, pdf
from .config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    try:
        yield
    finally:
        db.close_db()


app = FastAPI(title="rag-pgvector", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    document_id: int | None = None


class Source(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    chunker: chunking.Chunker
    chunks: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    chunker: chunking.Chunker = Form(chunking.Chunker.recursive),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    data = await file.read()
    text = pdf.extract_text(data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text in PDF")

    chunks = chunking.split(text, chunker)
    if not chunks:
        raise HTTPException(status_code=400, detail="PDF produced no chunks")

    vectors = embeddings.embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")

    pool = db.get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename, chunker) VALUES (%s, %s) RETURNING id",
                (file.filename, chunker.value),
            )
            row = cur.fetchone()
            assert row is not None
            doc_id: int = row[0]

            cur.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (doc_id, i, chunk, vec)
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                ],
            )
        conn.commit()

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        chunker=chunker,
        chunks=len(chunks),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    k = req.top_k or settings.top_k
    [qvec] = embeddings.embed_texts([req.question])

    pool = db.get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if req.document_id is not None:
                cur.execute(
                    "SELECT 1 FROM documents WHERE id = %s",
                    (req.document_id,),
                )
                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document {req.document_id} not found",
                    )

            cur.execute(
                """
                SELECT c.id, c.document_id, d.filename, c.chunk_index,
                       c.content, c.embedding <=> %s::vector AS distance
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE %s::int IS NULL OR c.document_id = %s::int
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (qvec, req.document_id, req.document_id, qvec, k),
            )
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No matching chunks found",
        )

    contexts = [r[4] for r in rows]
    answer = llm.answer_with_context(req.question, contexts)

    sources = [
        Source(
            chunk_id=r[0],
            document_id=r[1],
            filename=r[2],
            chunk_index=r[3],
            distance=float(r[5]),
        )
        for r in rows
    ]
    return QueryResponse(answer=answer, sources=sources)
