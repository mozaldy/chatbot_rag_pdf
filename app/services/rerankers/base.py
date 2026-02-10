from abc import ABC, abstractmethod
from typing import Sequence

from llama_index.core.schema import NodeWithScore


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[NodeWithScore],
        top_k: int,
        min_score: float | None = None,
    ) -> list[NodeWithScore]:
        """Return top-k reranked candidates."""
