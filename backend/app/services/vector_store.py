import json
from pathlib import Path
from typing import Any, Dict, List

from ..core.config import get_settings


class VectorStoreError(RuntimeError):
    """Raised when vector store operations fail."""


class ChromaVectorStore:
    """Persistent ChromaDB-backed storage for website chunks."""

    def __init__(
        self,
        db_dir: str | Path | None = None,
        collection_name: str | None = None,
    ):
        settings = get_settings()
        self.provider = "chroma"
        self.db_dir = Path(db_dir or settings.chroma_db_dir)
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import chromadb
        except Exception as exc:
            raise VectorStoreError("chromadb is required for the Chroma vector store.") from exc

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_dir))
        return self._client

    def get_or_create_collection(self):
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> dict:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk and embedding counts differ: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )
        if not chunks:
            return {
                "chunks_upserted": 0,
                "collection_name": self.collection_name,
                "vector_db_provider": self.provider,
            }

        ids = []
        documents = []
        metadatas = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise VectorStoreError("Each chunk must be a dictionary")
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            text = str(chunk.get("text") or "").strip()
            if not chunk_id:
                raise VectorStoreError("Chunk is missing chunk_id")
            if not text:
                raise VectorStoreError(f"Chunk {chunk_id} is missing text")
            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(flatten_chunk_metadata(chunk))

        self.get_or_create_collection().upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return {
            "chunks_upserted": len(chunks),
            "collection_name": self.collection_name,
            "vector_db_provider": self.provider,
        }

    def query_similar(
        self,
        query_embedding: list[float],
        website_id: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        if not website_id:
            raise VectorStoreError("website_id is required for vector retrieval")
        if not query_embedding:
            raise VectorStoreError("query_embedding is required for vector retrieval")

        where = _build_where_filter(website_id, filters or {})
        try:
            result = self.get_or_create_collection().query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k or 5)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        return _query_result_to_records(result)

    def delete_website_chunks(self, website_id: str) -> int:
        if not website_id:
            return 0
        existing_count = self.count_website_chunks(website_id)
        if existing_count:
            self.get_or_create_collection().delete(where={"website_id": website_id})
        return existing_count

    def count_website_chunks(self, website_id: str) -> int:
        if not website_id:
            return 0
        result = self.get_or_create_collection().get(where={"website_id": website_id}, include=[])
        return len(result.get("ids", []) or [])

    def get_website_chunks(
        self,
        website_id: str,
        limit: int = 2000,
        chunk_types: list[str] | None = None,
    ) -> list[dict]:
        """Return stored chunks for lexical reranking, filtered by website_id."""
        if not website_id:
            return []

        filters = {}
        if chunk_types:
            filters["chunk_type"] = [chunk_type for chunk_type in chunk_types if chunk_type]
        where = _build_where_filter(website_id, filters)
        result = self.get_or_create_collection().get(
            where=where,
            limit=max(1, int(limit or 2000)),
            include=["documents", "metadatas"],
        )
        return _get_result_to_records(result)

    def get_collection_stats(self) -> dict:
        collection = self.get_or_create_collection()
        return {
            "vector_db_provider": self.provider,
            "collection_name": self.collection_name,
            "db_dir": str(self.db_dir),
            "total_chunks": collection.count(),
        }


def flatten_chunk_metadata(chunk: dict) -> dict:
    """Flatten chunk metadata into Chroma-compatible scalar values."""
    metadata = {}
    chunk_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    combined = {
        **chunk_metadata,
        "website_id": chunk.get("website_id", ""),
        "source_url": chunk.get("source_url", ""),
        "page_title": chunk.get("page_title", ""),
        "page_type": chunk.get("page_type", ""),
        "chunk_type": chunk.get("chunk_type", ""),
        "chunk_index": chunk.get("chunk_index", 0),
    }

    for key, value in combined.items():
        clean_key = str(key)
        metadata[clean_key] = _metadata_value(value)
    return metadata


def _metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _build_where_filter(website_id: str, filters: Dict[str, Any]) -> dict:
    conditions: List[dict] = [{"website_id": website_id}]
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            values = [_metadata_value(item) for item in value if item is not None]
            if values:
                conditions.append({key: {"$in": values}})
        else:
            conditions.append({key: _metadata_value(value)})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _query_result_to_records(result: dict) -> list[dict]:
    ids = _first_result_list(result.get("ids"))
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    distances = _first_result_list(result.get("distances"))

    records = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        distance = distances[index] if index < len(distances) else None
        records.append(
            {
                "chunk_id": chunk_id,
                "text": documents[index] if index < len(documents) else "",
                "distance": distance,
                "score": _distance_to_score(distance),
                "source_url": metadata.get("source_url", ""),
                "page_title": metadata.get("page_title", ""),
                "chunk_type": metadata.get("chunk_type", ""),
                "metadata": metadata,
            }
        )
    return records


def _get_result_to_records(result: dict) -> list[dict]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    records = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        records.append(
            {
                "chunk_id": chunk_id,
                "text": documents[index] if index < len(documents) else "",
                "distance": None,
                "score": None,
                "source_url": metadata.get("source_url", ""),
                "page_title": metadata.get("page_title", ""),
                "chunk_type": metadata.get("chunk_type", ""),
                "metadata": metadata,
            }
        )
    return records


def _first_result_list(value: Any) -> list:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _distance_to_score(distance: Any) -> float | None:
    try:
        parsed = float(distance)
    except (TypeError, ValueError):
        return None
    return round(1.0 - parsed, 6)
