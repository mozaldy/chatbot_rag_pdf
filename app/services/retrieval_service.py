import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.core.llm_setup import get_embedding_model, get_fast_llm, get_llm, get_sparse_embedding_functions
from app.services.conversation_utils import (
    conversation_context_for_prompt,
    latest_user_query,
    normalize_conversation_messages,
)
from app.services.query_rewrite import normalize_query_text, rewrite_query_for_retrieval
from app.services.rag_quality import (
    build_context_and_sources,
    diversify_nodes_by_doc,
    fuse_hybrid_results,
    reciprocal_rank_fusion_ranked,
)
from app.services.rerankers import get_reranker
from app.services.rerankers.lexical import LexicalReranker
from app.services.query_router import get_query_router, RouteIntention


logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class RetrievalService:
    _shared_qdrant_client: AsyncQdrantClient | None = None
    _qdrant_client_lock: asyncio.Lock | None = None

    @classmethod
    def _get_qdrant_client_lock(cls) -> asyncio.Lock:
        if cls._qdrant_client_lock is None:
            cls._qdrant_client_lock = asyncio.Lock()
        return cls._qdrant_client_lock

    @classmethod
    async def get_qdrant_client(cls) -> AsyncQdrantClient:
        if cls._shared_qdrant_client is not None:
            return cls._shared_qdrant_client

        async with cls._get_qdrant_client_lock():
            if cls._shared_qdrant_client is None:
                cls._shared_qdrant_client = AsyncQdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=settings.QDRANT_TIMEOUT,
                )
        return cls._shared_qdrant_client

    @classmethod
    async def close_qdrant_client(cls) -> None:
        async with cls._get_qdrant_client_lock():
            if cls._shared_qdrant_client is None:
                return
            client = cls._shared_qdrant_client
            cls._shared_qdrant_client = None
        try:
            await asyncio.wait_for(client.close(), timeout=2.0)
        except Exception as exc:
            logger.debug("Failed to close shared AsyncQdrantClient cleanly: %s", exc)

    async def answer_query(self, query: str | Sequence[Any]) -> Dict[str, Any]:
        conversation_messages = normalize_conversation_messages(
            query,
            max_messages=settings.CONVERSATION_HISTORY_MAX_MESSAGES,
        )
        user_query = latest_user_query(conversation_messages)
        if not user_query:
            return {"response": "Please provide a non-empty question.", "sources": []}
        conversation_context = conversation_context_for_prompt(conversation_messages)

        timings: Dict[str, float] = {}
        start_total = time.perf_counter()
        qdrant_client = await self.get_qdrant_client()
        timings["conversation_messages"] = len(conversation_messages)
        timings["conversation_context_chars"] = len(conversation_context)
        timings["citation_retry"] = 0
        timings["citation_valid"] = 0

        try:
            stage_start = time.perf_counter()
            qdrant_ready, readiness_message = await self._check_qdrant_ready(qdrant_client)
            timings["qdrant_check_ms"] = self._elapsed_ms(stage_start)
            if not qdrant_ready:
                timings["total_ms"] = self._elapsed_ms(start_total)
                self._log_timings(timings)
                return {"response": readiness_message, "sources": []}

            stage_start = time.perf_counter()
            
            # --- QUERY ROUTING ---
            router = get_query_router()
            # Pass conversation history (excluding the current query if it's already in messages)
            # conversation_messages includes the current query at the end if normalize_conversation_messages appended it.
            # We want history *before* the current query.
            history_for_router = conversation_messages[:-1] if conversation_messages else []
            route_result = await router.route(user_query, chat_history=history_for_router)
            
            logger.info(f"Query Routing Result: {route_result.model_dump()}")

            # Prioritize Clarification if present, even if intention is not AMBIGUOUS
            if route_result.intention == RouteIntention.AMBIGUOUS or (route_result.clarification_question and len(route_result.clarification_question) > 5):
                timings["total_ms"] = self._elapsed_ms(start_total)
                self._log_timings(timings)
                return {
                    "response": route_result.clarification_question or "Mohon maaf, pertanyaan Anda kurang jelas. Bisa tolong diperjelas?",
                    "sources": [],
                }

            if route_result.intention == RouteIntention.CHIT_CHAT:
                # Handle chit-chat directly with LLM (no RAG)
                llm = get_fast_llm()
                prompt = (
                    "You are a helpful assistant for a document knowledge base. "
                    "Engage in the conversation politely and concisely. "
                    "Do not mention technical terms like 'LLM', 'RAG', 'AI model', or 'vector database'. "
                    "Act as a friendly guide helping the user find information.\n"
                    "Use Bahasa Indonesia for the response.\n"
                    f"User: {user_query}\n"
                    "Assistant:"
                )
                completion = await llm.acomplete(prompt)
                timings["total_ms"] = self._elapsed_ms(start_total)
                self._log_timings(timings)
                return {
                    "response": str(completion).strip(),
                    "sources": [],
                }

            # For SEARCH / SUMMARIZATION, use the rewritten query
            retrieval_query = route_result.rewritten_query or user_query
            
            # Safety checks for bad rewrites (e.g. "kesimpulan" -> "kesimpulan")
            # If the rewritten query is just a generic word, fall back to the user query (or extract topic from it if possible)
            # But simplistic fallback is safest.
            generic_keywords = {"summary", "summarize", "conclusion", "ringkasan", "kesimpulan", "rangkuman"}
            if retrieval_query.strip().lower() in generic_keywords:
                 logger.warning(
                     "Router produced generic query '%s'. Reverting to user query '%s'.", 
                     retrieval_query, 
                     user_query
                 )
                 retrieval_query = user_query
            
            # Optional: If enabling context history rewrite on top of router (usually not needed if router is smart)
            # But let's respect the router's rewrite as definitive for now.
            
            timings["conversation_ms"] = self._elapsed_ms(stage_start)

            stage_start = time.perf_counter()
            rewrite_result = await rewrite_query_for_retrieval(retrieval_query)
            timings["rewrite_ms"] = self._elapsed_ms(stage_start)
            retrieval_queries = rewrite_result.retrieval_queries or [retrieval_query]
            retrieval_weights = rewrite_result.retrieval_weights or [1.0]

            embed_model = get_embedding_model()
            sparse_doc_fn, sparse_query_fn = get_sparse_embedding_functions()
            vector_store = QdrantVectorStore(
                aclient=qdrant_client,
                collection_name=settings.COLLECTION_NAME,
                enable_hybrid=True,
                sparse_doc_fn=sparse_doc_fn,
                sparse_query_fn=sparse_query_fn,
                hybrid_fusion_fn=self._hybrid_fusion,
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model,
            )
            retriever = index.as_retriever(
                vector_store_query_mode="hybrid",
                similarity_top_k=settings.DENSE_TOP_K,
                sparse_top_k=settings.SPARSE_TOP_K,
                alpha=settings.HYBRID_ALPHA,
            )

            stage_start = time.perf_counter()
            candidate_lists: list[list[NodeWithScore]] = []
            for retrieval_query in retrieval_queries:
                retrieved = await retriever.aretrieve(retrieval_query)
                candidate_lists.append(list(retrieved[: settings.RERANK_CANDIDATES_K]))
            timings["retrieve_ms"] = self._elapsed_ms(stage_start)
            timings["retrieval_queries"] = len(retrieval_queries)

            stage_start = time.perf_counter()
            initial_candidates = self._merge_retrieval_candidates(
                candidate_lists=candidate_lists,
                weights=retrieval_weights,
                top_k=settings.RERANK_CANDIDATES_K,
            )
            timings["merge_ms"] = self._elapsed_ms(stage_start)
            timings["initial_candidates"] = len(initial_candidates)

            stage_start = time.perf_counter()
            reranker = get_reranker()
            try:
                reranked = await asyncio.to_thread(
                    reranker.rerank,
                    rewrite_result.query_for_rerank,
                    initial_candidates,
                    settings.RERANK_TOP_K,
                    settings.RERANK_MIN_SCORE,
                )
            except Exception as exc:
                logger.warning(
                    "Reranker '%s' failed (%s). Falling back to lexical reranker.",
                    settings.RERANKER_TYPE,
                    exc,
                )
                fallback = LexicalReranker()
                reranked = await asyncio.to_thread(
                    fallback.rerank,
                    rewrite_result.query_for_rerank,
                    initial_candidates,
                    settings.RERANK_TOP_K,
                    settings.RERANK_MIN_SCORE,
                )
            timings["rerank_ms"] = self._elapsed_ms(stage_start)

            stage_start = time.perf_counter()
            if settings.TABLE_PARENT_EXPANSION_ENABLED:
                reranked = await self._expand_table_parent_candidates(
                    qdrant_client=qdrant_client,
                    candidates=reranked,
                )
            timings["parent_expand_ms"] = self._elapsed_ms(stage_start)
            timings["reranked_candidates"] = len(reranked)
            timings["parent_expanded_candidates"] = len(reranked)

            if settings.RETRIEVAL_DEBUG_LOGS:
                logger.info(
                    "retrieval debug: queries=%s weights=%s initial=%s reranked=%s",
                    retrieval_queries,
                    retrieval_weights,
                    self._compact_candidates(initial_candidates),
                    self._compact_candidates(reranked),
                )

            if settings.CONTEXT_DIVERSITY_ENABLED:
                context_nodes = diversify_nodes_by_doc(
                    nodes=reranked,
                    max_items=settings.MAX_SOURCES,
                    max_per_doc=settings.MAX_SOURCES_PER_DOC,
                )
            else:
                context_nodes = list(reranked[: settings.MAX_SOURCES])
            timings["context_candidates"] = len(context_nodes)

            stage_start = time.perf_counter()
            context_str, sources = build_context_and_sources(
                nodes=context_nodes,
                max_chunk_chars=settings.MAX_CONTEXT_CHARS_PER_CHUNK,
                max_sources=settings.MAX_SOURCES,
                keep_full_table_parents=settings.TABLE_PARENT_ALWAYS_FULL_CONTEXT,
            )
            timings["context_ms"] = self._elapsed_ms(stage_start)
            timings["sources"] = len(sources)

            if not sources:
                timings["total_ms"] = self._elapsed_ms(start_total)
                self._log_timings(timings)
                
                # Dynamic "No Results" response in Bahasa Indonesia
                llm = get_fast_llm()
                prompt = (
                    "User asked: '{user_query}'\n"
                    "I searched the knowledge base but found no relevant documents.\n"
                    "Generate a polite apology in Bahasa Indonesia stating that no information was found "
                    "and suggesting they try a different query.\n"
                    "Do NOT start with 'Tentu', 'Baik', 'Mohon maaf', or 'Berikut'. Just say 'Saya tidak dapat menemukan info...' directly."
                )
                try:
                    completion = await llm.acomplete(prompt)
                    response_text = str(completion).strip()
                except Exception:
                    response_text = "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen."

                return {
                    "response": response_text,
                    "sources": [],
                }

            llm = get_llm()
            prompt = (
                "You are a retrieval-grounded QA assistant. Use only the provided sources.\n"
                "You are allowed to make basic logical deductions and temporal inferences STRICTLY based on the provided context. "
                "If a number or fact is presented next to references like 'increased from [Previous Year]' or accompanied by a chart caption indicating a specific year (e.g., 'Data for 2023'), "
                "you MUST deduce that the data belongs to that current timeframe/year. Do not consider this hallucination. "
                "You must still cite the source ID [Sx] where you drew this logical conclusion.\n"
                "Rules:\n"
                "1) If evidence is insufficient, say you don't have enough information.\n"
                "2) Cite every factual claim with source IDs like [S1] [S2].\n"
                "3) Do not cite sources that are not in the provided context.\n"
                "4) Keep the answer concise but complete.\n"
                "5) ANSWER IN BAHASA INDONESIA.\n"
                "6) Do not start with 'Tentu', 'Baik', 'Berikut adalah', or similar fillers. Answer directly.\n\n"
                f"Conversation context (may help resolve references):\n{conversation_context or 'N/A'}\n\n"
                f"Question:\n{user_query}\n\n"
                f"Sources:\n{context_str}\n\n"
                "Answer:"
            )

            stage_start = time.perf_counter()
            completion = await llm.acomplete(prompt)
            answer = str(completion).strip()

            if settings.REQUIRE_VALID_CITATIONS and sources:
                citation_ids = self._extract_citation_ids(answer)
                if (
                    not self._citation_ids_are_valid(citation_ids, sources)
                    and not self._is_insufficient_evidence_answer(answer)
                ):
                    timings["citation_retry"] = 1
                    logger.info(
                        "Regenerating response due to missing/invalid citations. cited_ids=%s source_ids=%s",
                        sorted(citation_ids),
                        sorted(self._source_ids(sources)),
                    )
                    retry_prompt = (
                        "You are a retrieval-grounded QA assistant. Use only the provided sources.\n"
                        "Rules:\n"
                        "1) Cite every factual claim with source IDs like [S1] [S2].\n"
                        "2) You may only cite from this allowed set: "
                        f"{', '.join(sorted(self._source_ids(sources)))}.\n"
                        "3) If evidence is insufficient, say you don't have enough information.\n"
                        "4) Do not provide uncited factual claims.\n\n"
                        f"Conversation context (may help resolve references):\n{conversation_context or 'N/A'}\n\n"
                        f"Question:\n{user_query}\n\n"
                        f"Sources:\n{context_str}\n\n"
                        "Answer:"
                    )
                    completion = await llm.acomplete(retry_prompt)
                    answer = str(completion).strip()
                    citation_ids = self._extract_citation_ids(answer)

                    if (
                        not self._citation_ids_are_valid(citation_ids, sources)
                        and not self._is_insufficient_evidence_answer(answer)
                    ):
                        answer = (
                            "I do not have enough grounded information to answer with valid source citations."
                        )
                        citation_ids = set()

                timings["citation_valid"] = int(
                    self._citation_ids_are_valid(citation_ids, sources)
                    or self._is_insufficient_evidence_answer(answer)
                )
            timings["generate_ms"] = self._elapsed_ms(stage_start)

            timings["total_ms"] = self._elapsed_ms(start_total)
            self._log_timings(timings)
            return {"response": answer, "sources": sources}
        except Exception:
            timings["total_ms"] = self._elapsed_ms(start_total)
            self._log_timings(timings)
            raise

    @staticmethod
    async def _check_qdrant_ready(qdrant_client: AsyncQdrantClient) -> tuple[bool, str]:
        try:
            collection_exists = await qdrant_client.collection_exists(
                collection_name=settings.COLLECTION_NAME
            )
        except Exception as exc:
            logger.warning("Qdrant readiness check failed: %s", exc)
            return (
                False,
                (
                    "Vector database is unreachable. Start Qdrant with "
                    "`docker-compose up -d` and retry."
                ),
            )

        if not collection_exists:
            return (
                False,
                (
                    f"No indexed documents found in collection '{settings.COLLECTION_NAME}'. "
                    "Upload and ingest a PDF first."
                ),
            )
        return True, ""

    @staticmethod
    async def _standalone_retrieval_query(
        user_query: str,
        conversation_messages: Sequence[dict[str, str]],
    ) -> str:
        if not settings.CONVERSATION_STANDALONE_QUERY_ENABLED:
            return user_query

        context = conversation_context_for_prompt(conversation_messages)
        if not context:
            return user_query

        prompt = (
            "Rewrite the user's latest question into a standalone retrieval query.\n"
            "Keep critical named entities, product names, and technical terms.\n"
            "Return one single line only, no explanation.\n\n"
            f"Conversation:\n{context}\n\n"
            f"Latest user question:\n{user_query}\n\n"
            "Standalone retrieval query:"
        )
        try:
            llm = get_fast_llm()
            completion = await llm.acomplete(prompt)
            rewritten = normalize_query_text(str(completion))
            if not rewritten:
                return user_query
            return rewritten[: settings.CONVERSATION_STANDALONE_MAX_CHARS]
        except Exception as exc:
            logger.warning("Standalone query generation failed; using latest user query: %s", exc)
            return user_query

    @staticmethod
    def _hybrid_fusion(
        dense_result: VectorStoreQueryResult,
        sparse_result: VectorStoreQueryResult,
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> VectorStoreQueryResult:
        return fuse_hybrid_results(
            dense_result=dense_result,
            sparse_result=sparse_result,
            top_k=top_k,
        )

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000.0, 2)

    @staticmethod
    def _extract_citation_ids(answer: str) -> set[str]:
        return {match for match in _CITATION_PATTERN.findall(answer or "")}

    @staticmethod
    def _source_ids(sources: Sequence[Dict[str, Any]]) -> set[str]:
        return {str(source.get("id")) for source in sources if source.get("id")}

    @classmethod
    def _citation_ids_are_valid(
        cls,
        citation_ids: set[str],
        sources: Sequence[Dict[str, Any]],
    ) -> bool:
        if not citation_ids:
            return False
        source_ids = cls._source_ids(sources)
        return bool(source_ids) and citation_ids.issubset(source_ids)

    @staticmethod
    def _is_insufficient_evidence_answer(answer: str) -> bool:
        normalized = (answer or "").strip().lower()
        if not normalized:
            return False
        markers = (
            "don't have enough information",
            "do not have enough information",
            "insufficient evidence",
            "not enough information",
            "tidak memiliki informasi",
            "tidak cukup informasi",
            "tidak ada informasi",
            "maaf", # Be careful with this, but usually "Maaf, saya tidak..." indicates failure in this context
            "kurang informasi",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _compact_candidates(candidates: Sequence[NodeWithScore], limit: int = 8) -> list[dict]:
        compact: list[dict] = []
        for candidate in list(candidates)[:limit]:
            metadata = candidate.node.metadata or {}
            compact.append(
                {
                    "node_id": candidate.node.node_id,
                    "doc_id": metadata.get("source_doc_id"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_kind": metadata.get("chunk_kind") or metadata.get("content_type"),
                    "table_id": metadata.get("table_id"),
                    "score": round(candidate.score, 6)
                    if candidate.score is not None
                    else None,
                }
            )
        return compact

    @staticmethod
    def _log_timings(timings: Dict[str, float]) -> None:
        logger.info(
            "chat pipeline timings ms: qdrant_check=%s conversation=%s rewrite=%s retrieve=%s merge=%s rerank=%s parent_expand=%s context=%s generate=%s total=%s mode=%s alpha=%s reranker=%s rewrite_enabled=%s context_diversity=%s require_valid_citations=%s standalone_query=%s",
            timings.get("qdrant_check_ms", 0.0),
            timings.get("conversation_ms", 0.0),
            timings.get("rewrite_ms", 0.0),
            timings.get("retrieve_ms", 0.0),
            timings.get("merge_ms", 0.0),
            timings.get("rerank_ms", 0.0),
            timings.get("parent_expand_ms", 0.0),
            timings.get("context_ms", 0.0),
            timings.get("generate_ms", 0.0),
            timings.get("total_ms", 0.0),
            settings.FUSION_MODE,
            settings.HYBRID_ALPHA,
            settings.RERANKER_TYPE,
            settings.QUERY_REWRITE_ENABLED,
            settings.CONTEXT_DIVERSITY_ENABLED,
            settings.REQUIRE_VALID_CITATIONS,
            settings.CONVERSATION_STANDALONE_QUERY_ENABLED,
        )
        logger.info(
            "chat pipeline counters: conversation_messages=%s context_chars=%s retrieval_queries=%s initial_candidates=%s reranked_candidates=%s parent_expanded_candidates=%s context_candidates=%s sources=%s citation_retry=%s citation_valid=%s",
            int(timings.get("conversation_messages", 0)),
            int(timings.get("conversation_context_chars", 0)),
            int(timings.get("retrieval_queries", 0)),
            int(timings.get("initial_candidates", 0)),
            int(timings.get("reranked_candidates", 0)),
            int(timings.get("parent_expanded_candidates", 0)),
            int(timings.get("context_candidates", 0)),
            int(timings.get("sources", 0)),
            int(timings.get("citation_retry", 0)),
            int(timings.get("citation_valid", 0)),
        )

    async def _expand_table_parent_candidates(
        self,
        qdrant_client: AsyncQdrantClient,
        candidates: Sequence[NodeWithScore],
    ) -> list[NodeWithScore]:
        if not candidates:
            return []

        base_candidates: list[NodeWithScore] = []
        table_parent_keys: set[tuple[str, str]] = set()
        anchor_by_parent: dict[tuple[str, str], NodeWithScore] = {}
        seen_node_ids: set[str] = set()

        for candidate in candidates:
            metadata = candidate.node.metadata or {}
            chunk_kind = str(metadata.get("chunk_kind") or metadata.get("content_type") or "")
            doc_id = str(metadata.get("source_doc_id") or "")
            table_id = str(metadata.get("table_id") or metadata.get("parent_id") or "")

            if chunk_kind == "table_parent" and doc_id and table_id:
                table_parent_keys.add((doc_id, table_id))

            if chunk_kind == "table_anchor" and doc_id:
                parent_id = str(metadata.get("parent_id") or table_id)
                if parent_id:
                    key = (doc_id, parent_id)
                    existing = anchor_by_parent.get(key)
                    if existing is None or (candidate.score or 0.0) > (existing.score or 0.0):
                        anchor_by_parent[key] = candidate
                    continue

            node_id = candidate.node.node_id
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            base_candidates.append(candidate)

        missing_parent_keys = [
            key for key in anchor_by_parent.keys() if key not in table_parent_keys
        ]

        if missing_parent_keys:
            fetch_tasks = []
            for doc_id, table_id in missing_parent_keys:
                anchor_candidate = anchor_by_parent[(doc_id, table_id)]
                seed_score = (anchor_candidate.score or 0.0) + 1e-6
                fetch_tasks.append(
                    self._fetch_table_parent_candidate(
                        qdrant_client=qdrant_client,
                        doc_id=doc_id,
                        table_id=table_id,
                        seed_score=seed_score,
                    )
                )
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for key, result in zip(missing_parent_keys, fetch_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to expand table parent for %s/%s: %s",
                        key[0],
                        key[1],
                        result,
                    )
                    continue
                if result is None:
                    continue
                node_id = result.node.node_id
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)
                base_candidates.append(result)
                table_parent_keys.add(key)

        for key, anchor_candidate in anchor_by_parent.items():
            if key in table_parent_keys:
                continue
            node_id = anchor_candidate.node.node_id
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            base_candidates.append(anchor_candidate)

        def _sort_key(item: NodeWithScore) -> float:
            if item.score is None:
                return float("-inf")
            return float(item.score)

        base_candidates.sort(key=_sort_key, reverse=True)
        return base_candidates

    async def _fetch_table_parent_candidate(
        self,
        qdrant_client: AsyncQdrantClient,
        doc_id: str,
        table_id: str,
        seed_score: float,
    ) -> NodeWithScore | None:
        scroll_filter = Filter(
            must=[
                FieldCondition(key="source_doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="chunk_kind", match=MatchValue(value="table_parent")),
                FieldCondition(key="table_id", match=MatchValue(value=table_id)),
            ]
        )
        records, _offset = await qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        node = self._record_to_text_node(records[0])
        if node is None:
            return None
        return NodeWithScore(node=node, score=seed_score)

    @staticmethod
    def _record_to_text_node(record: Any) -> TextNode | None:
        payload = getattr(record, "payload", None) or {}
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"_node_content", "_node_type"}
        }

        node_id = str(getattr(record, "id", metadata.get("chunk_id") or "unknown"))
        text = ""
        node_content = payload.get("_node_content")

        if isinstance(node_content, str):
            try:
                parsed = json.loads(node_content)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                parsed_text = parsed.get("text")
                if isinstance(parsed_text, str) and parsed_text.strip():
                    text = parsed_text.strip()
                parsed_id = parsed.get("id_")
                if parsed_id:
                    node_id = str(parsed_id)
                parsed_meta = parsed.get("metadata")
                if isinstance(parsed_meta, dict):
                    for key, value in parsed_meta.items():
                        metadata.setdefault(key, value)

        if not text:
            for key in ("text", "document", "content"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break

        if not text:
            return None

        return TextNode(id_=node_id, text=text, metadata=metadata)

    @staticmethod
    def _merge_retrieval_candidates(
        candidate_lists: list[list[NodeWithScore]],
        weights: list[float],
        top_k: int,
    ) -> list[NodeWithScore]:
        if not candidate_lists:
            return []
        if len(candidate_lists) == 1:
            return candidate_lists[0][:top_k]
        return reciprocal_rank_fusion_ranked(
            ranked_lists=candidate_lists,
            top_k=top_k,
        )
