# rag-pgvector

A small FastAPI service that ingests PDFs, chunks and embeds them, stores the vectors in Postgres (pgvector), and answers questions via retrieval-augmented generation. All inference (embeddings + chat) runs locally through [LM Studio](https://lmstudio.ai/)'s OpenAI-compatible API.

## Stack

- **FastAPI** — HTTP API
- **pgvector/pgvector:pg17** — vector store, HNSW index with cosine distance
- **LM Studio** — local OpenAI-compatible server for embeddings and chat
- **LangChain** — `RecursiveCharacterTextSplitter` and `SemanticChunker` (selectable per upload)
- **uv** — Python package and project manager

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (install: `winget install --id=astral-sh.uv` on Windows, `brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or any Docker daemon with Compose v2)
- [LM Studio](https://lmstudio.ai/) with two models downloaded:
  - An embedding model — default `text-embedding-nomic-embed-text-v1.5` (768 dims)can you add a README on how to set this project up
  - A chat model — default `openai/gpt-oss-20b`

## Setup

### 1. Start LM Studio's local server

Open LM Studio → **Developer** tab → load both models → click **Start Server**. The default endpoint is `http://localhost:1234/v1`. Confirm both models appear under *"Loaded"*.

### 2. Start Postgres

```bash
docker-compose up -d
```

This launches `pgvector/pgvector:pg17` on `localhost:5432` with credentials `rag` / `rag` and database `rag`. Data persists in the `pgdata` Docker volume.

### 3. Configure the app

```bash
cp .env.example .env
```

Edit `.env` if you're using different model names, ports, or chunk sizes. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Matches `docker-compose.yml` |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio's OpenAI endpoint |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5` | Must be loaded in LM Studio |
| `EMBEDDING_DIM` | `768` | Must match the model's output dimension |
| `CHAT_MODEL` | `openai/gpt-oss-20b` | Must be loaded in LM Studio |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Used by the recursive splitter |
| `TOP_K` | `5` | Default number of retrieved chunks |

> **If you change `EMBEDDING_DIM`** (e.g. switching to a 1024-dim model like `bge-m3`), drop the existing `chunks` table or wipe the `pgdata` volume — the column type `vector(N)` is fixed at table creation.

### 4. Install dependencies and run

```bash
uv sync
uv run uvicorn app.main:app --reload
```

The API is now at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

The `documents` and `chunks` tables, the `vector` extension, and the HNSW index are created automatically on first startup.

## API

### `POST /upload`

Multipart upload. Form fields:

- `file` (required) — a PDF.
- `chunker` (optional, default `recursive`) — `recursive` or `semantic`.

```bash
# default recursive chunker
curl -F "file=@manual.pdf" http://localhost:8000/upload

# semantic chunker (slower; embeds every sentence to find topic shifts)
curl -F "file=@manual.pdf" -F "chunker=semantic" http://localhost:8000/upload
```

Response:
```json
{ "document_id": 1, "filename": "manual.pdf", "chunker": "recursive", "chunks": 42 }
```

### `POST /query`

JSON body:

- `question` (required)
- `top_k` (optional) — overrides the default
- `document_id` (optional) — restricts retrieval to a single document

```bash
# search across everything
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"how do I reset the device?"}'

# only this document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"how do I reset the device?","document_id":1}'
```

Response includes the LLM's answer and the source chunks that were retrieved (with cosine distance).

### `GET /health`

Returns `{"status":"ok"}`. Useful as a liveness probe.

## How it works

1. **PDF text extraction** with `pypdf`.
2. **Chunking** — either:
   - `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap, prefers paragraph → newline → space → char boundaries), or
   - `SemanticChunker` (embeds each sentence and inserts a chunk break at the 95th-percentile cosine-distance jump).
3. **Embedding** via LM Studio (batched 32 at a time).
4. **Storage** — chunk text + 768-dim vector in `chunks`, parent document metadata (filename, chunker used) in `documents`. HNSW index with `vector_cosine_ops`.
5. **Retrieval** — embed the query, run kNN with `<=>` (cosine distance), optionally filtered by `document_id`.
6. **Generation** — concatenate the retrieved chunks as context and ask `openai/gpt-oss-20b` to answer using only that context.

## Common issues

**`No matching chunks found` on `/query`** — you haven't uploaded anything yet, or the `document_id` filter excludes everything.

**`operator does not exist: vector <=> double precision[]`** — the query parameter wasn't cast to `vector`. Already fixed via `%s::vector` in the query, but if you write new queries, remember to cast.

**Upload hangs / very slow with `chunker=semantic`** — expected; semantic chunking embeds every sentence individually before the per-chunk embeddings. Recursive is much faster.

**Embedding dimension mismatch on insert** — your loaded LM Studio embedding model produces a different dim than `EMBEDDING_DIM`. Update `.env` and recreate the `chunks` table (or wipe the volume).

**Connection refused to LM Studio** — confirm the server is running in LM Studio (Developer tab → "Start Server") and that `LMSTUDIO_BASE_URL` matches its actual port.

## Project layout

```
app/
  main.py        # FastAPI routes (/upload, /query, /health)
  config.py      # pydantic-settings, reads .env
  db.py          # psycopg pool, schema migrations, pgvector adapter
  embeddings.py  # LM Studio embeddings client (batched)
  llm.py         # LM Studio chat completion + RAG prompt
  pdf.py         # pypdf text extraction
  chunking.py    # recursive + semantic splitter selection
docker-compose.yml
pyproject.toml
.env.example
```
