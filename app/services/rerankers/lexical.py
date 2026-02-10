from typing import Sequence

from llama_index.core.schema import NodeWithScore

from app.services.rag_quality import rerank_nodes_by_query
from app.services.rerankers.base import BaseReranker


class LexicalReranker(BaseReranker):
    def __init__(self, lexical_weight: float = 0.35, rrf_k: int = 60) -> None:
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k

    def rerank(
        self,
        query: str,
        candidates: Sequence[NodeWithScore],
        top_k: int,
        min_score: float | None = None,
    ) -> list[NodeWithScore]:
        reranked = rerank_nodes_by_query(
            query=query,
            candidates=candidates,
            top_k=top_k,
            lexical_weight=self.lexical_weight,
            k=self.rrf_k,
        )
        if min_score is None:
            return reranked
        return [node for node in reranked if (node.score is not None and node.score >= min_score)]
