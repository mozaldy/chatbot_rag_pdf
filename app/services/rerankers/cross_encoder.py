import logging
from typing import Sequence

from llama_index.core.schema import NodeWithScore

from app.services.rerankers.base import BaseReranker


logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        use_fp16: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self._reranker = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[NodeWithScore],
        top_k: int,
        min_score: float | None = None,
    ) -> list[NodeWithScore]:
        if not candidates:
            return []

        reranker = self._get_or_load_model()
        pairs = [(query, candidate.node.get_content() or "") for candidate in candidates]
        scores = reranker.compute_score(
            pairs,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        if isinstance(scores, (float, int)):
            score_list = [float(scores)]
        else:
            score_list = [float(score) for score in scores]

        ranked = []
        for candidate, score in zip(candidates, score_list):
            ranked.append(NodeWithScore(node=candidate.node, score=score))

        ranked.sort(key=lambda item: item.score if item.score is not None else float("-inf"), reverse=True)
        if min_score is not None:
            ranked = [item for item in ranked if item.score is not None and item.score >= min_score]
        return ranked[:top_k]

    def _get_or_load_model(self):
        if self._reranker is not None:
            return self._reranker

        try:
            from FlagEmbedding import FlagReranker
        except Exception as exc:
            raise RuntimeError(
                "FlagEmbedding is required for cross_encoder reranker."
            ) from exc

        logger.info("Loading cross-encoder reranker model: %s", self.model_name)
        self._reranker = FlagReranker(
            self.model_name,
            use_fp16=self.use_fp16,
            normalize=True,
        )
        return self._reranker
