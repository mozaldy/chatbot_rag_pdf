import hashlib
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryResult


from app.core.config import settings


def normalize_markdown_text(text: str, preserve_page_markers: bool = False) -> str:
    """Normalize extracted markdown for better chunking and retrieval quality."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")  # soft hyphen

    # Join words split at line breaks by PDF hyphenation artifacts.
    normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)

    lines = normalized.split("\n")
    cleaned_lines = _drop_probable_boilerplate(
        lines,
        preserve_page_markers=preserve_page_markers,
    )
    normalized = "\n".join(cleaned_lines)

    # Normalize horizontal whitespace and blank line runs.
    # Preserve table alignment: don't collapse spaces within pipe-delimited lines.
    norm_lines = normalized.split("\n")
    for i, line in enumerate(norm_lines):
        if not line.strip().startswith("|"):
            norm_lines[i] = re.sub(r"[ \t]{2,}", " ", line)
    normalized = "\n".join(norm_lines)
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
) -> List[NodeWithScore]:
    """Fuse ranked candidate lists with Standard Reciprocal Rank Fusion."""
    fused: Dict[str, Tuple[NodeWithScore, float]] = {}
    if not ranked_lists:
        return []

    # Use k from settings, avoiding hardcoded values
    k = settings.FUSION_RRF_K

    for ranked_nodes in ranked_lists:
        for rank, node_with_score in enumerate(ranked_nodes, start=1):
            node_id = node_with_score.node.node_id
            # Standard RRF formula: 1 / (k + rank)
            # No weights applied, treating all lists equally.
            score = 1.0 / (k + rank)
            
            if node_id in fused:
                fused[node_id] = (fused[node_id][0], fused[node_id][1] + score)
            else:
                fused[node_id] = (node_with_score, score)

    merged = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:top_k]
    return [NodeWithScore(node=item[0].node, score=item[1]) for item in merged]


def fuse_hybrid_results(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    top_k: int,
) -> VectorStoreQueryResult:
    """Fuse dense and sparse retrieval results using standard RRF."""
    dense_ranked = _vector_result_to_ranked(dense_result)
    sparse_ranked = _vector_result_to_ranked(sparse_result)

    # Simplified fusion: always use standard RRF without weights
    fused_ranked = reciprocal_rank_fusion_ranked(
        ranked_lists=[dense_ranked, sparse_ranked],
        top_k=top_k,
    )
    fused_nodes = [node_with_score.node for node_with_score in fused_ranked]
    fused_scores = [node_with_score.score for node_with_score in fused_ranked]
    fused_ids = [node.node_id for node in fused_nodes]
    return VectorStoreQueryResult(nodes=fused_nodes, similarities=fused_scores, ids=fused_ids)


def rerank_nodes_by_query(
    candidates: Sequence[NodeWithScore],
    top_k: int,
) -> List[NodeWithScore]:
    """
    Sort candidates purely by their existing score (e.g. from Cross-Encoder).
    Removes manual lexical blending to preserve probability calibration.
    """
    if not candidates:
        return []

    # Sort purely by the reranker's score (descending)
    # If score is None, we treat it as -infinity or handle gracefully, 
    # but here we assume reranker provides scores.
    sorted_candidates = sorted(
        candidates, 
        key=lambda node: node.score if node.score is not None else -1.0, 
        reverse=True
    )

    return list(sorted_candidates[:top_k])


def diversify_nodes_by_doc(
    nodes: Sequence[NodeWithScore],
    max_items: int,
    max_per_doc: int,
) -> List[NodeWithScore]:
    """
    Select ranked nodes while limiting how many chunks come from one document.
    A second pass fills remaining slots if strict caps remove too many items.
    """
    if not nodes or max_items <= 0:
        return []

    if max_per_doc <= 0:
        return _dedupe_nodes(nodes, max_items)

    selected: List[NodeWithScore] = []
    selected_ids: set[str] = set()
    per_doc_counts: dict[str, int] = defaultdict(int)

    for candidate in nodes:
        node_id = candidate.node.node_id
        if node_id in selected_ids:
            continue
        doc_key = _doc_key(candidate)
        if per_doc_counts[doc_key] >= max_per_doc:
            continue
        selected.append(candidate)
        selected_ids.add(node_id)
        per_doc_counts[doc_key] += 1
        if len(selected) >= max_items:
            return selected

    if len(selected) >= max_items:
        return selected

    for candidate in nodes:
        node_id = candidate.node.node_id
        if node_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(node_id)
        if len(selected) >= max_items:
            break

    return selected


def build_context_and_sources(
    nodes: Sequence[NodeWithScore],
    max_chunk_chars: int,
    max_sources: int,
    keep_full_table_parents: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build source-labeled context blocks and response source payload."""
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, node_with_score in enumerate(nodes[:max_sources], start=1):
        source_id = f"S{idx}"
        metadata = node_with_score.node.metadata or {}
        full_text = (node_with_score.node.get_content() or "").strip()

        content_type = metadata.get("content_type", "text")
        chunk_kind = metadata.get("chunk_kind") or content_type
        if keep_full_table_parents and chunk_kind == "table_parent":
            snippet = full_text
        else:
            snippet = _trim_text(full_text, max_chunk_chars)
        source = {
            "id": source_id,
            "filename": metadata.get("filename", "unknown"),
            "doc_id": metadata.get("source_doc_id", "unknown"),
            "chunk_id": metadata.get("chunk_id", node_with_score.node.node_id),
            "chunk_index": metadata.get("chunk_index", "?"),
            "page_label": metadata.get("page_label"),
            "section_title": metadata.get("section_title"),
            "score": round(node_with_score.score, 6)
            if node_with_score.score is not None
            else None,
            "text": snippet,
            "node_id": node_with_score.node.node_id,
            "content_type": content_type,
            "chunk_kind": chunk_kind,
            "parent_id": metadata.get("parent_id"),
            "table_id": metadata.get("table_id"),
            "schema_version": metadata.get("schema_version"),
            "table_visual_status": metadata.get("table_visual_status"),
        }
        sources.append(source)

        context_blocks.append(
            "\n".join(
                [
                    f"[{source_id}]",
                    f"filename: {source['filename']}",
                    f"content_type: {content_type}",
                    f"chunk_kind: {source['chunk_kind'] or 'unknown'}",
                    f"table_id: {source['table_id'] or 'n/a'}",
                    f"parent_id: {source['parent_id'] or 'n/a'}",
                    f"doc_id: {source['doc_id']}",
                    f"chunk_id: {source['chunk_id']}",
                    f"chunk_index: {source['chunk_index']}",
                    f"page_label: {source['page_label'] or 'unknown'}",
                    f"section_title: {source['section_title'] or 'unknown'}",
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


def _dedupe_nodes(nodes: Sequence[NodeWithScore], max_items: int) -> List[NodeWithScore]:
    unique: List[NodeWithScore] = []
    seen_ids: set[str] = set()
    for candidate in nodes:
        node_id = candidate.node.node_id
        if node_id in seen_ids:
            continue
        unique.append(candidate)
        seen_ids.add(node_id)
        if len(unique) >= max_items:
            break
    return unique


def _doc_key(candidate: NodeWithScore) -> str:
    metadata = candidate.node.metadata or {}
    doc_id = metadata.get("source_doc_id")
    if doc_id:
        return str(doc_id)
    return f"__node__:{candidate.node.node_id}"


def _drop_probable_boilerplate(
    lines: Iterable[str],
    preserve_page_markers: bool = False,
) -> List[str]:
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
        # Never drop markdown table rows.
        if stripped.startswith("|"):
            return False
        lowered = stripped.casefold()

        if not preserve_page_markers and _looks_like_page_marker(lowered):
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





def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..."
