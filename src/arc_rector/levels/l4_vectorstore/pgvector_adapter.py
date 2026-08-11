"""L4 -- vector database, Postgres + pgvector implementation.

Same contract as every other L4 adapter: hold chunk vectors, return the nearest
ones. The appeal of pgvector is that it is not a separate service -- vectors sit
in the same database as the rest of an application's data, in a plain table you
can join against.

Three gotchas:

1. `<=>` is cosine *distance* (0 = identical), so the score returned here is
   `1 - distance`, matching the cosine similarity every other adapter reports.
2. The ORDER BY must be on the raw distance, ascending. Ordering by the
   converted score descending is mathematically identical but the planner will
   not use the ivfflat/hnsw index for it, silently turning every search into a
   sequential scan.
3. pgvector cannot index a `vector` column wider than 2000 dimensions, so the
   index step is skipped above that (search still works, just unindexed).

`CREATE EXTENSION vector` needs the extension files installed on the server and
a role allowed to create extensions; the pgvector/pgvector Docker image gives
both. Targets psycopg 3.3.4 (`pip install "psycopg[binary]"`).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ...interfaces import VectorStore
from ...registry import require
from ...types import Chunk, Retrieved

_PIP_NAME = "psycopg[binary]"
_ADAPTER = "pgvector"
_MAX_INDEXABLE_DIM = 2000


class PgVectorStore(VectorStore):
    """Postgres-backed vector store using the pgvector extension."""

    name = "pgvector"

    def __init__(
        self,
        *,
        dsn: str = "postgresql://arc:arc@localhost:5433/arc",
        url: str = "",
        table: str = "arc_chunks",
        index_type: str = "hnsw",
        lists: int = 100,
        m: int = 16,
        ef_construction: int = 64,
        connect_timeout: int = 10,
        **_: Any,
    ) -> None:
        self.dsn = url or dsn  # `url` is the generic config.yaml key
        self.table = table
        self.index_type = index_type.lower()
        self.lists = int(lists)
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.connect_timeout = int(connect_timeout)
        self._conn: Any = None

    # -- connection ---------------------------------------------------------

    @property
    def _sql(self) -> Any:
        """`psycopg.sql` is a submodule: importing `psycopg` alone may not bind it."""
        return require(_PIP_NAME, _ADAPTER, "psycopg.sql")

    @property
    def _client(self) -> Any:
        """Connect on first use, so merely selecting this adapter never fails."""
        if self._conn is None or getattr(self._conn, "closed", False):
            psycopg = require(_PIP_NAME, _ADAPTER, "psycopg")
            self._conn = psycopg.connect(
                self.dsn, autocommit=True, connect_timeout=self.connect_timeout
            )
        return self._conn

    # -- VectorStore --------------------------------------------------------

    def ensure_collection(self, dim: int) -> None:
        sql = self._sql
        table = sql.Identifier(self.table)
        with self._client.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {table} ("
                    "  chunk_id  TEXT PRIMARY KEY,"
                    "  embedding vector({dim}) NOT NULL,"
                    "  payload   JSONB NOT NULL"
                    ")"
                ).format(table=table, dim=sql.Literal(int(dim)))
            )
            if int(dim) <= _MAX_INDEXABLE_DIM:
                cur.execute(self._index_statement(dim))

    def _index_statement(self, dim: int) -> Any:
        sql = self._sql
        fields = {
            "index": sql.Identifier(f"{self.table}_embedding_idx"),
            "table": sql.Identifier(self.table),
        }
        if self.index_type == "ivfflat":
            return sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {table}"
                " USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            ).format(lists=sql.Literal(self.lists), **fields)
        return sql.SQL(
            "CREATE INDEX IF NOT EXISTS {index} ON {table}"
            " USING hnsw (embedding vector_cosine_ops)"
            " WITH (m = {m}, ef_construction = {ef_construction})"
        ).format(
            m=sql.Literal(self.m),
            ef_construction=sql.Literal(self.ef_construction),
            **fields,
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        sql = self._sql
        statement = sql.SQL(
            "INSERT INTO {table} (chunk_id, embedding, payload)"
            " VALUES (%s, %s::vector, %s::jsonb)"
            " ON CONFLICT (chunk_id) DO UPDATE"
            "   SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload"
        ).format(table=sql.Identifier(self.table))
        rows = [
            (chunk.chunk_id, self._vector_literal(vector), json.dumps(chunk.payload(), default=str))
            for chunk, vector in zip(chunks, vectors)
        ]
        with self._client.cursor() as cur:
            cur.executemany(statement, rows)
        return len(rows)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[Retrieved]:
        if not self._table_exists():
            return []
        sql = self._sql
        statement = sql.SQL(
            "SELECT payload, 1 - (embedding <=> %s::vector) AS score"
            " FROM {table}"
            " ORDER BY embedding <=> %s::vector"  # raw distance ASC keeps the index in play
            " LIMIT %s"
        ).format(table=sql.Identifier(self.table))
        literal = self._vector_literal(vector)
        with self._client.cursor() as cur:
            cur.execute(statement, (literal, literal, max(1, int(top_k))))
            rows = cur.fetchall()
        return [
            Retrieved(chunk=Chunk.from_payload(self._as_dict(payload)), score=float(score))
            for payload, score in rows
        ]

    def count(self) -> int:
        if not self._table_exists():
            return 0
        sql = self._sql
        with self._client.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {table}").format(
                table=sql.Identifier(self.table)
            ))
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def drop(self) -> None:
        sql = self._sql
        with self._client.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=sql.Identifier(self.table)
            ))

    def close(self) -> None:
        if self._conn is not None and not getattr(self._conn, "closed", False):
            self._conn.close()
        self._conn = None

    # -- helpers ------------------------------------------------------------

    def _table_exists(self) -> bool:
        """Lets count()/search() answer honestly before the first ingest."""
        with self._client.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (self.table,))
            row = cur.fetchone()
        return bool(row and row[0])

    @staticmethod
    def _vector_literal(vector: Sequence[float]) -> str:
        """pgvector's text input format: '[1.0,2.0,3.0]' -- repr() round-trips floats."""
        return "[" + ",".join(repr(float(value)) for value in vector) + "]"

    @staticmethod
    def _as_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, (str, bytes)):
            return json.loads(payload)
        return {}
