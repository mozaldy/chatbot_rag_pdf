from typing import Any, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings


class DocumentLifecycleService:
    def __init__(self, client: QdrantClient | None = None):
        self.qdrant_client = client or QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT,
        )

    def list_documents(self, max_points: int | None = None) -> Dict[str, Any]:
        limit = max_points or settings.DOCUMENT_LIST_MAX_POINTS
        page_size = min(settings.DOCUMENT_LIST_PAGE_SIZE, limit)
        docs, scanned_points, truncated = self._collect_documents(
            scroll_filter=None,
            max_points=limit,
            page_size=page_size,
        )
        documents = sorted(
            docs.values(),
            key=lambda item: (item["filename"].lower(), item["doc_id"]),
        )
        total_chunks = sum(item["chunks"] for item in documents)
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "scanned_points": scanned_points,
            "truncated": truncated,
        }

    def get_document(self, doc_id: str) -> Dict[str, Any] | None:
        docs, _scanned_points, _truncated = self._collect_documents(
            scroll_filter=Filter(
                must=[FieldCondition(key="source_doc_id", match=MatchValue(value=doc_id))]
            ),
            max_points=settings.DOCUMENT_LIST_MAX_POINTS,
            page_size=settings.DOCUMENT_LIST_PAGE_SIZE,
        )
        return docs.get(doc_id)

    def delete_by_doc_id(self, doc_id: str) -> Dict[str, Any]:
        deleted_chunks = self._delete_by_filter(
            key="source_doc_id",
            value=doc_id,
        )
        if deleted_chunks == 0:
            return {
                "status": "not_found",
                "message": f"No indexed chunks found for doc_id '{doc_id}'.",
                "deleted_chunks": 0,
                "doc_id": doc_id,
                "filename": None,
            }
        return {
            "status": "deleted",
            "message": f"Deleted {deleted_chunks} chunks for doc_id '{doc_id}'.",
            "deleted_chunks": deleted_chunks,
            "doc_id": doc_id,
            "filename": None,
        }

    def delete_by_filename(self, filename: str) -> Dict[str, Any]:
        deleted_chunks = self._delete_by_filter(
            key="filename",
            value=filename,
        )
        if deleted_chunks == 0:
            return {
                "status": "not_found",
                "message": f"No indexed chunks found for filename '{filename}'.",
                "deleted_chunks": 0,
                "doc_id": None,
                "filename": filename,
            }
        return {
            "status": "deleted",
            "message": f"Deleted {deleted_chunks} chunks for filename '{filename}'.",
            "deleted_chunks": deleted_chunks,
            "doc_id": None,
            "filename": filename,
        }

    def _delete_by_filter(self, key: str, value: str) -> int:
        if not self.qdrant_client.collection_exists(collection_name=settings.COLLECTION_NAME):
            return 0

        selector_filter = Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value))]
        )
        existing = self.qdrant_client.count(
            collection_name=settings.COLLECTION_NAME,
            count_filter=selector_filter,
            exact=True,
        )
        delete_count = int(existing.count)
        if delete_count == 0:
            return 0

        self.qdrant_client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=selector_filter,
            wait=True,
        )
        return delete_count

    def _collect_documents(
        self,
        scroll_filter: Filter | None,
        max_points: int,
        page_size: int,
    ) -> tuple[Dict[str, Dict[str, Any]], int, bool]:
        if not self.qdrant_client.collection_exists(collection_name=settings.COLLECTION_NAME):
            return {}, 0, False

        documents: Dict[str, Dict[str, Any]] = {}
        scanned_points = 0
        truncated = False
        offset = None

        while True:
            remaining = max_points - scanned_points
            if remaining <= 0:
                truncated = True
                break

            records, next_offset = self.qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=min(page_size, remaining),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break

            for record in records:
                self._accumulate_record(documents, record.payload or {})

            scanned_points += len(records)
            if next_offset is None:
                break
            offset = next_offset

        cleaned_documents = {}
        for doc_id, item in documents.items():
            chunk_kind_counts = dict(sorted(item["chunk_kind_counts"].items()))
            cleaned = {
                "doc_id": item["source_doc_id"],
                "filename": item["filename"],
                "chunks": item["chunks"],
                "max_chunk_index": item["max_chunk_index"],
                "table_parent_chunks": chunk_kind_counts.get("table_parent", 0),
                "table_anchor_chunks": chunk_kind_counts.get("table_anchor", 0),
                "chunk_kind_counts": chunk_kind_counts,
                "schema_versions": sorted(item["schema_versions"]),
            }
            cleaned_documents[doc_id] = cleaned

        return cleaned_documents, scanned_points, truncated

    @staticmethod
    def _accumulate_record(
        documents: Dict[str, Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> None:
        doc_id = str(payload.get("source_doc_id", "unknown"))
        filename = str(payload.get("filename", "unknown"))
        chunk_index_raw = payload.get("chunk_index")
        chunk_index = chunk_index_raw if isinstance(chunk_index_raw, int) else None
        chunk_kind_raw = payload.get("chunk_kind") or payload.get("content_type") or "text"
        chunk_kind = str(chunk_kind_raw)
        schema_version = payload.get("schema_version")

        if doc_id not in documents:
            documents[doc_id] = {
                "source_doc_id": doc_id,
                "filename": filename,
                "chunks": 0,
                "max_chunk_index": None,
                "chunk_kind_counts": {},
                "schema_versions": set(),
            }

        target = documents[doc_id]
        target["chunks"] += 1
        target["chunk_kind_counts"][chunk_kind] = target["chunk_kind_counts"].get(chunk_kind, 0) + 1
        if isinstance(schema_version, int):
            target["schema_versions"].add(schema_version)
        if chunk_index is not None:
            current = target["max_chunk_index"]
            target["max_chunk_index"] = chunk_index if current is None else max(current, chunk_index)
