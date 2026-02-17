import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llama_index.core.schema import NodeWithScore, TextNode

from app.services.retrieval_service import RetrievalService


class RetrievalServiceCitationTests(unittest.TestCase):
    def test_extract_citation_ids_reads_unique_source_ids(self) -> None:
        answer = "Feature A is supported [S1]. Another claim [S2] and repeat [S1]."
        citations = RetrievalService._extract_citation_ids(answer)
        self.assertEqual(citations, {"S1", "S2"})

    def test_citation_ids_are_valid_only_when_subset_of_sources(self) -> None:
        sources = [{"id": "S1"}, {"id": "S2"}, {"id": "S3"}]
        self.assertTrue(RetrievalService._citation_ids_are_valid({"S1", "S3"}, sources))
        self.assertFalse(RetrievalService._citation_ids_are_valid({"S4"}, sources))
        self.assertFalse(RetrievalService._citation_ids_are_valid(set(), sources))

    def test_insufficient_evidence_answer_detection(self) -> None:
        self.assertTrue(
            RetrievalService._is_insufficient_evidence_answer(
                "I do not have enough information to answer this question."
            )
        )
        self.assertTrue(
            RetrievalService._is_insufficient_evidence_answer(
                "There is insufficient evidence in the provided context."
            )
        )
        self.assertFalse(
            RetrievalService._is_insufficient_evidence_answer(
                "The model uses OCR for plate recognition [S1]."
            )
        )


class RetrievalServiceTableExpansionTests(unittest.IsolatedAsyncioTestCase):
    async def test_expand_table_parent_candidates_replaces_anchor_with_parent(self) -> None:
        service = RetrievalService()
        anchor = TextNode(
            id_="anchor_1",
            text="anchor text",
            metadata={
                "source_doc_id": "doc_1",
                "chunk_kind": "table_anchor",
                "table_id": "table_0001",
                "parent_id": "table_0001",
            },
        )
        parent = TextNode(
            id_="parent_1",
            text="full table parent",
            metadata={
                "source_doc_id": "doc_1",
                "chunk_kind": "table_parent",
                "table_id": "table_0001",
            },
        )

        with patch.object(
            service,
            "_fetch_table_parent_candidate",
            new=AsyncMock(return_value=NodeWithScore(node=parent, score=0.8)),
        ):
            expanded = await service._expand_table_parent_candidates(
                qdrant_client=AsyncMock(),
                candidates=[NodeWithScore(node=anchor, score=0.7)],
            )

        kinds = [item.node.metadata.get("chunk_kind") for item in expanded]
        self.assertIn("table_parent", kinds)
        self.assertNotIn("table_anchor", kinds)

    async def test_expand_table_parent_candidates_keeps_anchor_when_parent_missing(self) -> None:
        service = RetrievalService()
        anchor = TextNode(
            id_="anchor_1",
            text="anchor text",
            metadata={
                "source_doc_id": "doc_1",
                "chunk_kind": "table_anchor",
                "table_id": "table_0001",
                "parent_id": "table_0001",
            },
        )

        with patch.object(
            service,
            "_fetch_table_parent_candidate",
            new=AsyncMock(return_value=None),
        ):
            expanded = await service._expand_table_parent_candidates(
                qdrant_client=AsyncMock(),
                candidates=[NodeWithScore(node=anchor, score=0.7)],
            )

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].node.metadata.get("chunk_kind"), "table_anchor")

    def test_record_to_text_node_parses_llama_payload(self) -> None:
        payload = {
            "chunk_kind": "table_parent",
            "table_id": "table_0001",
            "source_doc_id": "doc_1",
            "_node_content": json.dumps(
                {
                    "id_": "node_123",
                    "text": "| A | B |",
                    "metadata": {"chunk_kind": "table_parent", "table_id": "table_0001"},
                }
            ),
        }
        record = SimpleNamespace(id="point_1", payload=payload)
        node = RetrievalService._record_to_text_node(record)

        assert node is not None
        self.assertEqual(node.node_id, "node_123")
        self.assertEqual(node.get_content(), "| A | B |")
        self.assertEqual(node.metadata.get("chunk_kind"), "table_parent")


if __name__ == "__main__":
    unittest.main()
