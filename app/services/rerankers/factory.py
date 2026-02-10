import logging

from app.core.config import settings
from app.services.rerankers.base import BaseReranker
from app.services.rerankers.cross_encoder import CrossEncoderReranker
from app.services.rerankers.lexical import LexicalReranker


logger = logging.getLogger(__name__)
_RERANKER: BaseReranker | None = None
_RERANKER_KEY: str | None = None


def get_reranker() -> BaseReranker:
    global _RERANKER, _RERANKER_KEY

    key = (
        f"{settings.RERANKER_TYPE}|{settings.RERANKER_MODEL}|"
        f"{settings.RERANKER_BATCH_SIZE}|{settings.RERANKER_MAX_LENGTH}"
    )
    if _RERANKER is not None and _RERANKER_KEY == key:
        return _RERANKER

    reranker_type = settings.RERANKER_TYPE.strip().lower()
    if reranker_type == "lexical":
        _RERANKER = LexicalReranker(rrf_k=settings.FUSION_RRF_K)
    elif reranker_type in {"cross_encoder", "cross-encoder"}:
        try:
            _RERANKER = CrossEncoderReranker(
                model_name=settings.RERANKER_MODEL,
                use_fp16=settings.RERANKER_USE_FP16,
                batch_size=settings.RERANKER_BATCH_SIZE,
                max_length=settings.RERANKER_MAX_LENGTH,
            )
        except Exception as exc:
            logger.warning(
                "Cross-encoder reranker initialization failed (%s). Falling back to lexical reranker.",
                exc,
            )
            _RERANKER = LexicalReranker(rrf_k=settings.FUSION_RRF_K)
    else:
        logger.warning(
            "Unsupported RERANKER_TYPE=%s. Falling back to lexical reranker.",
            settings.RERANKER_TYPE,
        )
        _RERANKER = LexicalReranker(rrf_k=settings.FUSION_RRF_K)

    _RERANKER_KEY = key
    return _RERANKER
