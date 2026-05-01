import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def init_db() -> None:
    """Create the vector extension, open a pool, and ensure schema exists."""
    global pool

    # The vector type must exist before register_vector can introspect its OID.
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        open=True,
    )

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunker TEXT NOT NULL DEFAULT 'recursive',
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS chunker TEXT NOT NULL DEFAULT 'recursive';
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({settings.embedding_dim}) NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
        conn.commit()


def close_db() -> None:
    global pool
    if pool is not None:
        pool.close()
        pool = None


def get_pool() -> ConnectionPool:
    if pool is None:
        raise RuntimeError("Database pool is not initialized")
    return pool
