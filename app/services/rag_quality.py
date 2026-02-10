import hashlib
import re
import unicodedata
import uuid
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryResult


DEFAULT_RRF_K = 60


def normalize_markdown_text(text: str) -> str:
    """Normalize extracted markdown for better chunking and retrieval quality."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")  # soft hyphen

    # Join words split at line breaks by PDF hyphenation artifacts.
    normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)

    lines = normalized.split("\n")
    cleaned_lines = _drop_probable_boilerplate(lines)
    normalized = "\n".join(cleaned_lines)

    # Normalize horizontal whitespace and blank line runs.
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def compute_document_id(_filename: str, normalized_text: str) -> str:
    """Stable content-addressed document identifier."""
    payload = normalized_text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"doc_{digest[:20]}"


def compute_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:chunk:{chunk_index:05d}"


def compute_point_uuid(chunk_id: str) -> str:
    """Deterministic UUID for vector store point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def reciprocal_rank_fusion_ranked(
    ranked_lists: Sequence[Sequence[NodeWithScore]],
    top_k: int,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> List[NodeWithScore]:
    """Fuse ranked candidate lists with Reciprocal Rank Fusion."""
    fused: Dict[str, Tuple[NodeWithScore, float]] = {}
    if not ranked_lists:
        return []

    if weights is None:
        normalized_weights = [1.0] * len(ranked_lists)
    else:
        if len(weights) != len(ranked_lists):
            raise ValueError("weights length must match ranked_lists length")
        normalized_weights = [max(0.0, float(weight)) for weight in weights]
        if sum(normalized_weights) == 0:
            normalized_weights = [1.0] * len(ranked_lists)

    for list_idx, ranked_nodes in enumerate(ranked_lists):
        list_weight = normalized_weights[list_idx]
        for rank, node_with_score in enumerate(ranked_nodes, start=1):
            node_id = node_with_score.node.node_id
            score = list_weight / (rank + k)
            if node_id in fused:
                fused[node_id] = (fused[node_id][0], fused[node_id][1] + score)
            else:
                fused[node_id] = (node_with_score, score)

    merged = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:top_k]
    return [NodeWithScore(node=item[0].node, score=item[1]) for item in merged]


def fuse_hybrid_results(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float,
    top_k: int,
    mode: str = "weighted_rrf",
    k: int = DEFAULT_RRF_K,
) -> VectorStoreQueryResult:
    """Fuse dense and sparse retrieval results using configurable RRF variants."""
    dense_ranked = _vector_result_to_ranked(dense_result)
    sparse_ranked = _vector_result_to_ranked(sparse_result)
    fusion_mode = mode.strip().lower()

    if fusion_mode == "weighted_rrf":
        dense_weight = min(max(alpha, 0.0), 1.0)
        sparse_weight = 1.0 - dense_weight
        weights = [dense_weight, sparse_weight]
    elif fusion_mode == "rrf":
        weights = [1.0, 1.0]
    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")

    fused_ranked = reciprocal_rank_fusion_ranked(
        ranked_lists=[dense_ranked, sparse_ranked],
        top_k=top_k,
        k=k,
        weights=weights,
    )
    fused_nodes = [node_with_score.node for node_with_score in fused_ranked]
    fused_scores = [node_with_score.score for node_with_score in fused_ranked]
    fused_ids = [node.node_id for node in fused_nodes]
    return VectorStoreQueryResult(nodes=fused_nodes, similarities=fused_scores, ids=fused_ids)


def rerank_nodes_by_query(
    query: str,
    candidates: Sequence[NodeWithScore],
    top_k: int,
    lexical_weight: float = 0.35,
    k: int = DEFAULT_RRF_K,
) -> List[NodeWithScore]:
    """
    Re-rank candidates with a blend of rank signal and lexical overlap.
    Works with hybrid retrieval scores that are not 0-1 normalized.
    """
    if not candidates:
        return []

    query_terms = _query_terms(query)
    if not query_terms:
        return list(candidates[:top_k])

    scored: List[Tuple[NodeWithScore, float]] = []
    for rank, candidate in enumerate(candidates, start=1):
        rank_signal = 1.0 / (rank + k)
        chunk_text = candidate.node.get_content().lower()
        lexical = _lexical_overlap(query_terms, chunk_text)
        fused_score = (1 - lexical_weight) * rank_signal + lexical_weight * lexical
        scored.append((candidate, fused_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        NodeWithScore(node=item[0].node, score=item[1])
        for item in scored[:top_k]
    ]


def build_context_and_sources(
    nodes: Sequence[NodeWithScore],
    max_chunk_chars: int,
    max_sources: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build source-labeled context blocks and response source payload."""
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, node_with_score in enumerate(nodes[:max_sources], start=1):
        source_id = f"S{idx}"
        metadata = node_with_score.node.metadata or {}
        full_text = (node_with_score.node.get_content() or "").strip()
        snippet = _trim_text(full_text, max_chunk_chars)

        source = {
            "id": source_id,
            "filename": metadata.get("filename", "unknown"),
            "doc_id": metadata.get("doc_id", "unknown"),
            "chunk_id": metadata.get("chunk_id", node_with_score.node.node_id),
            "chunk_index": metadata.get("chunk_index", "?"),
            "page_label": metadata.get("page_label"),
            "score": round(node_with_score.score, 6)
            if node_with_score.score is not None
            else None,
            "text": snippet,
            "node_id": node_with_score.node.node_id,
        }
        sources.append(source)

        context_blocks.append(
            "\n".join(
                [
                    f"[{source_id}]",
                    f"filename: {source['filename']}",
                    f"doc_id: {source['doc_id']}",
                    f"chunk_id: {source['chunk_id']}",
                    f"chunk_index: {source['chunk_index']}",
                    "content:",
                    snippet,
                ]
            )
        )

    return "\n\n".join(context_blocks), sources


def _vector_result_to_ranked(result: VectorStoreQueryResult) -> List[NodeWithScore]:
    nodes = result.nodes or []
    similarities = result.similarities or []
    ranked: List[NodeWithScore] = []
    for idx, node in enumerate(nodes):
        score = similarities[idx] if idx < len(similarities) else None
        ranked.append(NodeWithScore(node=node, score=score))
    return ranked


def _drop_probable_boilerplate(lines: Iterable[str]) -> List[str]:
    stripped_lines = [line.strip() for line in lines]
    total = len(stripped_lines) or 1
    short_candidates = [
        line.casefold()
        for line in stripped_lines
        if line
        and len(line) <= 80
        and len(line.split()) <= 12
        and not _starts_with_markdown_syntax(line)
    ]
    counts = Counter(short_candidates)

    def should_drop(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        lowered = stripped.casefold()

        if _looks_like_page_marker(lowered):
            return True

        repeat_count = counts.get(lowered, 0)
        if repeat_count >= 4 and (repeat_count / total) >= 0.04:
            return True
        return False

    return [line for line in lines if not should_drop(line)]


def _starts_with_markdown_syntax(line: str) -> bool:
    return bool(re.match(r"^(#{1,6}\s|[-*+]\s|\d+\.\s|>\s|\|)", line.strip()))


def _looks_like_page_marker(lowered: str) -> bool:
    return bool(
        re.fullmatch(r"page\s*\d+(\s*(of|/)\s*\d+)?", lowered)
        or re.fullmatch(r"\d+\s*(of|/)\s*\d+", lowered)
        or re.fullmatch(r"\d{1,4}", lowered)
    )


def _query_terms(query: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]{2,}", query.lower())}


def _lexical_overlap(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in text)
    return hits / float(len(query_terms))


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..."
