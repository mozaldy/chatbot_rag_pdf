import asyncio
import unittest

from llama_index.core.schema import NodeWithScore, TextNode

from app.core.config import settings
from app.services.query_rewrite import normalize_query_text, rewrite_query_for_retrieval, rule_based_rewrite
from app.services.rerankers.factory import get_reranker
from app.services.rerankers.lexical import LexicalReranker


class QueryRewriteAndRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            "QUERY_REWRITE_ENABLED": settings.QUERY_REWRITE_ENABLED,
            "QUERY_REWRITE_MODE": settings.QUERY_REWRITE_MODE,
            "QUERY_REWRITE_WEIGHT": settings.QUERY_REWRITE_WEIGHT,
            "QUERY_REWRITE_MAX_TERMS": settings.QUERY_REWRITE_MAX_TERMS,
            "RERANKER_TYPE": settings.RERANKER_TYPE,
            "RERANKER_MODEL": settings.RERANKER_MODEL,
            "RERANKER_BATCH_SIZE": settings.RERANKER_BATCH_SIZE,
            "RERANKER_MAX_LENGTH": settings.RERANKER_MAX_LENGTH,
        }

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            setattr(settings, key, value)

    def test_normalize_query_text_collapses_whitespace(self) -> None:
        raw = "  what is\n\nthis\tquery about?  "
        self.assertEqual(normalize_query_text(raw), "what is this query about?")

    def test_rule_based_rewrite_filters_stopwords_and_deduplicates(self) -> None:
        query = "What are the traffic traffic analysis features in the application"
        rewritten = rule_based_rewrite(query)
        self.assertNotIn("what", rewritten.split())
        self.assertIn("traffic", rewritten.split())
        self.assertEqual(rewritten.split().count("traffic"), 1)

    def test_rewrite_query_disabled_uses_single_query(self) -> None:
        settings.QUERY_REWRITE_ENABLED = False

        result = asyncio.run(rewrite_query_for_retrieval("What is the model used?"))

        self.assertEqual(len(result.retrieval_queries), 1)
        self.assertEqual(result.retrieval_weights, [1.0])
        self.assertEqual(result.retrieval_queries[0], normalize_query_text("What is the model used?"))

    def test_rewrite_query_rule_mode_adds_rewritten_variant(self) -> None:
        settings.QUERY_REWRITE_ENABLED = True
        settings.QUERY_REWRITE_MODE = "rule"
        settings.QUERY_REWRITE_WEIGHT = 0.7

        result = asyncio.run(
            rewrite_query_for_retrieval(
                "What are the main features of the traffic analysis application?"
            )
        )

        self.assertEqual(len(result.retrieval_queries), 2)
        self.assertIsNotNone(result.rewritten_query)
        self.assertAlmostEqual(sum(result.retrieval_weights), 1.0, places=6)

    def test_lexical_reranker_applies_min_score_filter(self) -> None:
        reranker = LexicalReranker(lexical_weight=1.0, rrf_k=60)
        query = "vehicle plate numbers"
        candidates = [
            NodeWithScore(node=TextNode(id_="n1", text="General description only."), score=0.8),
            NodeWithScore(node=TextNode(id_="n2", text="The system reads vehicle plate numbers."), score=0.7),
        ]

        reranked = reranker.rerank(query=query, candidates=candidates, top_k=2, min_score=0.9)
        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0].node.node_id, "n2")

    def test_factory_falls_back_to_lexical_for_unknown_type(self) -> None:
        settings.RERANKER_TYPE = "unknown_mode"
        reranker = get_reranker()
        self.assertIsInstance(reranker, LexicalReranker)


if __name__ == "__main__":
    unittest.main()
