import unittest

from app.core.config import settings
from app.services.document_lifecycle_service import DocumentLifecycleService


class _FakeCountResult:
    def __init__(self, count: int):
        self.count = count


class _FakeRecord:
    def __init__(self, point_id: str, payload: dict):
        self.id = point_id
        self.payload = payload


class _FakeQdrantClient:
    def __init__(
        self,
        *,
        collection_exists: bool = True,
        scroll_pages: list[tuple[list[_FakeRecord], object | None]] | None = None,
        counts: dict[tuple[str, str], int] | None = None,
    ):
        self._collection_exists = collection_exists
        self._scroll_pages = list(scroll_pages or [])
        self._counts = counts or {}
        self.delete_calls = []

    def collection_exists(self, collection_name: str) -> bool:
        return self._collection_exists

    def scroll(self, **kwargs):
        if not self._scroll_pages:
            return [], None
        return self._scroll_pages.pop(0)

    def count(self, **kwargs):
        count_filter = kwargs["count_filter"]
        condition = count_filter.must[0]
        key = condition.key
        value = condition.match.value
        return _FakeCountResult(self._counts.get((key, value), 0))

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


class DocumentLifecycleServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_max = settings.DOCUMENT_LIST_MAX_POINTS
        self._saved_page_size = settings.DOCUMENT_LIST_PAGE_SIZE
        settings.DOCUMENT_LIST_MAX_POINTS = 100
        settings.DOCUMENT_LIST_PAGE_SIZE = 2

    def tearDown(self) -> None:
        settings.DOCUMENT_LIST_MAX_POINTS = self._saved_max
        settings.DOCUMENT_LIST_PAGE_SIZE = self._saved_page_size

    def test_list_documents_aggregates_chunks_per_document(self) -> None:
        client = _FakeQdrantClient(
            scroll_pages=[
                (
                    [
                        _FakeRecord("p1", {"doc_id": "doc_a", "filename": "a.pdf", "chunk_index": 0}),
                        _FakeRecord("p2", {"doc_id": "doc_a", "filename": "a.pdf", "chunk_index": 1}),
                    ],
                    "offset_1",
                ),
                (
                    [
                        _FakeRecord("p3", {"doc_id": "doc_b", "filename": "b.pdf", "chunk_index": 0}),
                    ],
                    None,
                ),
            ]
        )
        service = DocumentLifecycleService(client=client)

        result = service.list_documents(max_points=10)

        self.assertEqual(result["total_documents"], 2)
        self.assertEqual(result["total_chunks"], 3)
        self.assertEqual(result["scanned_points"], 3)
        self.assertFalse(result["truncated"])
        doc_a = next(doc for doc in result["documents"] if doc["doc_id"] == "doc_a")
        self.assertEqual(doc_a["chunks"], 2)
        self.assertEqual(doc_a["max_chunk_index"], 1)

    def test_list_documents_marks_truncated_when_max_points_reached(self) -> None:
        client = _FakeQdrantClient(
            scroll_pages=[
                (
                    [
                        _FakeRecord("p1", {"doc_id": "doc_a", "filename": "a.pdf", "chunk_index": 0}),
                        _FakeRecord("p2", {"doc_id": "doc_b", "filename": "b.pdf", "chunk_index": 0}),
                    ],
                    "offset_1",
                ),
                (
                    [
                        _FakeRecord("p3", {"doc_id": "doc_c", "filename": "c.pdf", "chunk_index": 0}),
                    ],
                    None,
                ),
            ]
        )
        service = DocumentLifecycleService(client=client)

        result = service.list_documents(max_points=2)

        self.assertTrue(result["truncated"])
        self.assertEqual(result["scanned_points"], 2)
        self.assertEqual(result["total_documents"], 2)

    def test_get_document_returns_none_if_not_found(self) -> None:
        client = _FakeQdrantClient(scroll_pages=[([], None)])
        service = DocumentLifecycleService(client=client)
        self.assertIsNone(service.get_document("missing_doc"))

    def test_get_document_returns_summary_when_found(self) -> None:
        client = _FakeQdrantClient(
            scroll_pages=[
                (
                    [
                        _FakeRecord("p1", {"doc_id": "doc_x", "filename": "x.pdf", "chunk_index": 0}),
                        _FakeRecord("p2", {"doc_id": "doc_x", "filename": "x.pdf", "chunk_index": 2}),
                    ],
                    None,
                )
            ]
        )
        service = DocumentLifecycleService(client=client)

        result = service.get_document("doc_x")

        assert result is not None
        self.assertEqual(result["doc_id"], "doc_x")
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["max_chunk_index"], 2)

    def test_delete_by_doc_id_returns_deleted_with_chunk_count(self) -> None:
        client = _FakeQdrantClient(counts={("doc_id", "doc_a"): 4})
        service = DocumentLifecycleService(client=client)

        result = service.delete_by_doc_id("doc_a")

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["deleted_chunks"], 4)
        self.assertEqual(len(client.delete_calls), 1)

    def test_delete_by_filename_returns_not_found_when_empty(self) -> None:
        client = _FakeQdrantClient(counts={})
        service = DocumentLifecycleService(client=client)

        result = service.delete_by_filename("x.pdf")

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["deleted_chunks"], 0)
        self.assertEqual(len(client.delete_calls), 0)

    def test_list_documents_returns_empty_when_collection_missing(self) -> None:
        client = _FakeQdrantClient(collection_exists=False)
        service = DocumentLifecycleService(client=client)

        result = service.list_documents(max_points=10)

        self.assertEqual(result["documents"], [])
        self.assertEqual(result["total_documents"], 0)
        self.assertEqual(result["total_chunks"], 0)


if __name__ == "__main__":
    unittest.main()
