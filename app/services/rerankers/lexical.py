from typing import Sequence

from llama_index.core.schema import NodeWithScore

from app.services.rag_quality import rerank_nodes_by_query
from app.services.rerankers.base import BaseReranker


class LexicalReranker(BaseReranker):
    def __init__(self) -> None:
        pass

    def rerank(
        self,
        query: str,
        candidates: Sequence[NodeWithScore],
        top_k: int,
        min_score: float | None = None,
    ) -> list[NodeWithScore]:
        reranked = rerank_nodes_by_query(
            candidates=candidates,
            top_k=top_k,
        )
        if min_score is None:
            return reranked
        return [node for node in reranked if (node.score is not None and node.score >= min_score)]
