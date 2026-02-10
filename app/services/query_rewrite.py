import logging
import re
import unicodedata
from dataclasses import dataclass

from app.core.config import settings
from app.core.llm_setup import get_fast_llm


logger = logging.getLogger(__name__)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


@dataclass
class QueryRewriteResult:
    original_query: str
    normalized_query: str
    rewritten_query: str | None
    retrieval_queries: list[str]
    retrieval_weights: list[float]

    @property
    def query_for_rerank(self) -> str:
        return self.original_query or self.normalized_query


async def rewrite_query_for_retrieval(query: str) -> QueryRewriteResult:
    normalized = normalize_query_text(query)
    if not normalized:
        return QueryRewriteResult(
            original_query=query,
            normalized_query="",
            rewritten_query=None,
            retrieval_queries=[],
            retrieval_weights=[],
        )

    if not settings.QUERY_REWRITE_ENABLED:
        return QueryRewriteResult(
            original_query=query,
            normalized_query=normalized,
            rewritten_query=None,
            retrieval_queries=[normalized],
            retrieval_weights=[1.0],
        )

    mode = settings.QUERY_REWRITE_MODE.strip().lower()
    rewritten: str | None = None
    if mode == "rule":
        rewritten = rule_based_rewrite(normalized)
    elif mode == "llm":
        rewritten = await llm_rewrite(normalized)
    else:
        logger.warning("Unsupported QUERY_REWRITE_MODE=%s; using original query.", settings.QUERY_REWRITE_MODE)

    queries = []
    weights = []
    rewrite_weight = min(max(float(settings.QUERY_REWRITE_WEIGHT), 0.0), 1.0)
    if rewritten and rewritten != normalized:
        queries.extend([rewritten, normalized])
        weights.extend([rewrite_weight, 1.0 - rewrite_weight])
    else:
        queries.append(normalized)
        weights.append(1.0)

    dedup_queries: list[str] = []
    dedup_weights: list[float] = []
    for text, weight in zip(queries, weights):
        if text in dedup_queries:
            idx = dedup_queries.index(text)
            dedup_weights[idx] += weight
        else:
            dedup_queries.append(text)
            dedup_weights.append(weight)

    return QueryRewriteResult(
        original_query=query,
        normalized_query=normalized,
        rewritten_query=rewritten if rewritten != normalized else None,
        retrieval_queries=dedup_queries,
        retrieval_weights=dedup_weights,
    )


def normalize_query_text(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def rule_based_rewrite(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9]{2,}", query.lower())
    if not tokens:
        return query

    filtered = [token for token in tokens if token not in _STOPWORDS]
    if not filtered:
        filtered = tokens

    unique_terms: list[str] = []
    for token in filtered:
        if token not in unique_terms:
            unique_terms.append(token)

    rewritten = " ".join(unique_terms[:settings.QUERY_REWRITE_MAX_TERMS]).strip()
    return rewritten or query


async def llm_rewrite(query: str) -> str | None:
    try:
        llm = get_fast_llm()
        prompt = (
            "Rewrite the user query for retrieval. Return a concise keyword-focused query.\n"
            "Keep named entities, product names, and technical terms.\n"
            "Return one line only, no explanation.\n\n"
            f"User query:\n{query}\n\n"
            "Rewritten query:"
        )
        completion = await llm.acomplete(prompt)
        rewritten = normalize_query_text(str(completion))
        if not rewritten:
            return None
        return rewritten[: settings.QUERY_REWRITE_MAX_CHARS]
    except Exception as exc:
        logger.warning("Query rewrite failed, using original query: %s", exc)
        return None
