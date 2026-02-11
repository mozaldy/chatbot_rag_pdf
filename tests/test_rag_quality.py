import unittest

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from app.services.rag_quality import (
    build_context_and_sources,
    compute_chunk_id,
    compute_document_id,
    compute_point_uuid,
    diversify_nodes_by_doc,
    fuse_hybrid_results,
    normalize_markdown_text,
    rerank_nodes_by_query,
)


class RagQualityUtilityTests(unittest.TestCase):
    def test_normalize_markdown_text_removes_hyphenation_and_page_markers(self) -> None:
        raw = (
            "Page 1 of 10\n"
            "This is a hyphen-\n"
            "ated word.\n\n"
            "Page 2 of 10\n"
            "Next line."
        )
        normalized = normalize_markdown_text(raw)
        self.assertIn("hyphenated", normalized)
        self.assertNotIn("Page 1 of 10", normalized)
        self.assertNotIn("Page 2 of 10", normalized)

    def test_document_and_chunk_ids_are_deterministic(self) -> None:
        text = "A stable document body."
        doc_id_1 = compute_document_id("Example.pdf", text)
        doc_id_2 = compute_document_id("Example.pdf", text)
        self.assertEqual(doc_id_1, doc_id_2)
        chunk_id = compute_chunk_id(doc_id_1, 7)
        self.assertEqual(chunk_id, f"{doc_id_1}:chunk:00007")
        point_uuid = compute_point_uuid(chunk_id)
        self.assertEqual(point_uuid, compute_point_uuid(chunk_id))
        self.assertEqual(len(point_uuid), 36)

    def test_rerank_nodes_promotes_lexically_relevant_chunks(self) -> None:
        query = "vehicle plate numbers"
        node_1 = TextNode(id_="n1", text="General introduction with no query terms.")
        node_2 = TextNode(
            id_="n2",
            text="The system reads vehicle plate numbers using OCR.",
        )

        candidates = [
            NodeWithScore(node=node_1, score=0.9),
            NodeWithScore(node=node_2, score=0.8),
        ]
        reranked = rerank_nodes_by_query(query=query, candidates=candidates, top_k=2)
        self.assertEqual(reranked[0].node.node_id, "n2")

    def test_build_context_and_sources_returns_stable_source_payload(self) -> None:
        node = TextNode(
            id_="doc_a:chunk:00000",
            text="Evidence text.",
            metadata={
                "filename": "a.pdf",
                "doc_id": "doc_a",
                "chunk_id": "doc_a:chunk:00000",
                "chunk_index": 0,
                "page_label": "Page 3",
                "section_title": "Capabilities",
            },
        )
        context, sources = build_context_and_sources(
            nodes=[NodeWithScore(node=node, score=0.42)],
            max_chunk_chars=1000,
            max_sources=3,
        )
        self.assertIn("[S1]", context)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["id"], "S1")
        self.assertEqual(sources[0]["doc_id"], "doc_a")
        self.assertEqual(sources[0]["chunk_id"], "doc_a:chunk:00000")
        self.assertEqual(sources[0]["page_label"], "Page 3")
        self.assertEqual(sources[0]["section_title"], "Capabilities")

    def test_fuse_hybrid_results_respects_alpha_for_weighted_rrf(self) -> None:
        dense_top = TextNode(id_="dense_top", text="dense")
        dense_second = TextNode(id_="dense_second", text="dense second")
        sparse_top = TextNode(id_="sparse_top", text="sparse")
        sparse_second = TextNode(id_="sparse_second", text="sparse second")

        dense_result = VectorStoreQueryResult(
            nodes=[dense_top, dense_second],
            similarities=[0.9, 0.8],
            ids=["dense_top", "dense_second"],
        )
        sparse_result = VectorStoreQueryResult(
            nodes=[sparse_top, sparse_second],
            similarities=[0.88, 0.77],
            ids=["sparse_top", "sparse_second"],
        )

        dense_weighted = fuse_hybrid_results(
            dense_result=dense_result,
            sparse_result=sparse_result,
            alpha=0.9,
            top_k=1,
            mode="weighted_rrf",
            k=60,
        )
        sparse_weighted = fuse_hybrid_results(
            dense_result=dense_result,
            sparse_result=sparse_result,
            alpha=0.1,
            top_k=1,
            mode="weighted_rrf",
            k=60,
        )

        self.assertEqual(dense_weighted.nodes[0].node_id, "dense_top")
        self.assertEqual(sparse_weighted.nodes[0].node_id, "sparse_top")

    def test_fuse_hybrid_results_rrf_mode_ignores_alpha(self) -> None:
        shared = TextNode(id_="shared", text="shared")
        dense_only = TextNode(id_="dense_only", text="dense")
        sparse_only = TextNode(id_="sparse_only", text="sparse")

        dense_result = VectorStoreQueryResult(
            nodes=[shared, dense_only],
            similarities=[0.9, 0.7],
            ids=["shared", "dense_only"],
        )
        sparse_result = VectorStoreQueryResult(
            nodes=[shared, sparse_only],
            similarities=[0.85, 0.75],
            ids=["shared", "sparse_only"],
        )

        high_alpha = fuse_hybrid_results(
            dense_result=dense_result,
            sparse_result=sparse_result,
            alpha=0.95,
            top_k=3,
            mode="rrf",
            k=60,
        )
        low_alpha = fuse_hybrid_results(
            dense_result=dense_result,
            sparse_result=sparse_result,
            alpha=0.05,
            top_k=3,
            mode="rrf",
            k=60,
        )

        self.assertEqual([node.node_id for node in high_alpha.nodes], [node.node_id for node in low_alpha.nodes])

    def test_diversify_nodes_by_doc_limits_single_document_dominance(self) -> None:
        ranked = [
            NodeWithScore(
                node=TextNode(id_="a1", text="a1", metadata={"doc_id": "doc_a"}),
                score=0.95,
            ),
            NodeWithScore(
                node=TextNode(id_="a2", text="a2", metadata={"doc_id": "doc_a"}),
                score=0.92,
            ),
            NodeWithScore(
                node=TextNode(id_="a3", text="a3", metadata={"doc_id": "doc_a"}),
                score=0.90,
            ),
            NodeWithScore(
                node=TextNode(id_="b1", text="b1", metadata={"doc_id": "doc_b"}),
                score=0.88,
            ),
            NodeWithScore(
                node=TextNode(id_="c1", text="c1", metadata={"doc_id": "doc_c"}),
                score=0.84,
            ),
        ]
        diversified = diversify_nodes_by_doc(
            nodes=ranked,
            max_items=4,
            max_per_doc=1,
        )
        self.assertEqual([node.node.node_id for node in diversified], ["a1", "b1", "c1", "a2"])

    def test_diversify_nodes_by_doc_deduplicates_when_no_limit(self) -> None:
        shared = TextNode(id_="shared", text="shared", metadata={"doc_id": "doc_a"})
        ranked = [
            NodeWithScore(node=shared, score=0.9),
            NodeWithScore(node=shared, score=0.8),
            NodeWithScore(node=TextNode(id_="b1", text="b1", metadata={"doc_id": "doc_b"}), score=0.7),
        ]
        diversified = diversify_nodes_by_doc(
            nodes=ranked,
            max_items=3,
            max_per_doc=0,
        )
        self.assertEqual([node.node.node_id for node in diversified], ["shared", "b1"])


if __name__ == "__main__":
    unittest.main()
