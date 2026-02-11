import asyncio
import os
import shutil
import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

from fastapi import UploadFile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.core.llm_setup import get_fast_llm, get_embedding_model, get_sparse_embedding_functions
from app.services.rag_quality import (
    compute_chunk_id,
    compute_document_id,
    compute_point_uuid,
    normalize_markdown_text,
)

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Import common API errors if possible to be specific, or generic Exception for now
try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

logger = logging.getLogger(__name__)
_MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$")
_PAGE_PATTERN = re.compile(r"^page\s*(\d+)(?:\s*(?:of|/)\s*(\d+))?$", re.IGNORECASE)
_PAGE_RATIO_PATTERN = re.compile(r"^(\d+)\s*(?:of|/)\s*(\d+)$", re.IGNORECASE)
_PAGE_NUMERIC_PATTERN = re.compile(r"^\d{1,4}$")

class IngestionService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT,
        )
        self.llm = get_fast_llm()
        self.embed_model = get_embedding_model()
        self.sparse_doc_fn, self.sparse_query_fn = get_sparse_embedding_functions()

    async def process_pdf(self, file: UploadFile):
        temp_path = self._save_upload_file(file)
        try:
            logger.info(f"Parsing PDF with Docling: {file.filename}")
            raw_markdown = await asyncio.to_thread(self._extract_markdown_from_pdf, temp_path)
            normalized_markdown = normalize_markdown_text(raw_markdown)
            page_aware_markdown = normalize_markdown_text(raw_markdown, preserve_page_markers=True)

            if not normalized_markdown:
                raise ValueError("No readable content extracted from PDF.")

            doc_id = compute_document_id(file.filename, normalized_markdown)
            existing_chunks = await asyncio.to_thread(self._count_chunks_for_doc, doc_id)
            if existing_chunks > 0:
                logger.info(
                    "Skipping duplicate ingestion for %s (doc_id=%s, chunks=%s)",
                    file.filename,
                    doc_id,
                    existing_chunks,
                )
                return {
                    "filename": file.filename,
                    "status": "duplicate_skipped",
                    "chunks": existing_chunks,
                    "global_summary": "Duplicate content already indexed.",
                    "doc_id": doc_id,
                    "replaced_points": 0,
                }

            replaced_points = await asyncio.to_thread(
                self._delete_points_for_filename,
                file.filename,
            )
            logger.info(
                "Prepared ingestion for %s (doc_id=%s, replaced_points=%s)",
                file.filename,
                doc_id,
                replaced_points,
            )

            logger.info("Generating global summary for %s", file.filename)
            global_summary = await self._generate_summary(normalized_markdown)

            valid_nodes = await asyncio.to_thread(
                self._build_nodes,
                normalized_markdown,
                page_aware_markdown,
                doc_id,
                file.filename,
            )

            logger.info(f"Upserting {len(valid_nodes)} nodes to Qdrant...")
            await asyncio.to_thread(self._upsert_nodes, valid_nodes)
            
            return {
                "filename": file.filename,
                "status": "ingested",
                "chunks": len(valid_nodes),
                "global_summary": global_summary,
                "doc_id": doc_id,
                "replaced_points": replaced_points,
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise e
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_upload_file(self, upload_file: UploadFile) -> str:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name

    def _extract_markdown_from_pdf(self, temp_path: str) -> str:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = settings.ENABLE_OCR
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(temp_path)
        return result.document.export_to_markdown()

    def _build_nodes(
        self,
        normalized_markdown: str,
        page_aware_markdown: str,
        doc_id: str,
        filename: str,
    ):
        parser = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
        )

        documents = [Document(
            text=normalized_markdown,
            id_=doc_id,
            metadata={
                "filename": filename,
                "doc_id": doc_id,
            }
        )]
        nodes = parser.get_nodes_from_documents(documents)
        if not nodes:
            raise ValueError("No chunks generated from parsed document.")

        section_events = self._extract_section_events(normalized_markdown)
        page_events = self._extract_page_events(page_aware_markdown)
        section_positions = self._align_node_positions(nodes, normalized_markdown)
        page_positions = self._align_node_positions(nodes, page_aware_markdown)

        valid_nodes = []
        for idx, node in enumerate(nodes):
            chunk_id = compute_chunk_id(doc_id, idx)
            node.id_ = compute_point_uuid(chunk_id)
            section_title = self._event_for_position(section_positions[idx], section_events)
            page_label = self._event_for_position(page_positions[idx], page_events)

            node.metadata["filename"] = filename
            node.metadata["doc_id"] = doc_id
            node.metadata["chunk_index"] = idx
            node.metadata["chunk_id"] = chunk_id
            node.metadata["total_chunks"] = len(nodes)
            node.metadata["section_title"] = section_title or "Document"
            if page_label:
                node.metadata["page_label"] = page_label
            node.excluded_embed_metadata_keys.extend(
                [
                    "filename",
                    "doc_id",
                    "chunk_index",
                    "chunk_id",
                    "total_chunks",
                    "section_title",
                    "page_label",
                ]
            )
            valid_nodes.append(node)

        return valid_nodes

    def _extract_section_events(self, text: str) -> list[tuple[int, str]]:
        events: list[tuple[int, str]] = [(0, "Document")]
        cursor = 0
        for line in text.splitlines(keepends=True):
            match = _MARKDOWN_HEADING_PATTERN.match(line.strip())
            if match:
                title = match.group(1).strip()
                if title:
                    events.append((cursor, title[:200]))
            cursor += len(line)
        return events

    def _extract_page_events(self, text: str) -> list[tuple[int, str]]:
        events: list[tuple[int, str]] = []
        cursor = 0
        for line in text.splitlines(keepends=True):
            page_label = self._canonical_page_label(line.strip())
            if page_label:
                events.append((cursor, page_label))
            cursor += len(line)
        return events

    def _canonical_page_label(self, value: str) -> str | None:
        if not value:
            return None

        match = _PAGE_PATTERN.fullmatch(value)
        if match:
            return f"Page {int(match.group(1))}"

        match = _PAGE_RATIO_PATTERN.fullmatch(value)
        if match:
            return f"Page {int(match.group(1))}"

        if _PAGE_NUMERIC_PATTERN.fullmatch(value):
            number = int(value)
            if 1 <= number <= 5000:
                return f"Page {number}"
        return None

    def _align_node_positions(
        self,
        nodes: Sequence,
        text: str,
    ) -> list[int | None]:
        positions: list[int | None] = []
        cursor = 0
        for node in nodes:
            content = (node.get_content() or "").strip()
            pos = self._find_chunk_position(content, text, cursor)
            positions.append(pos)
            if pos is not None:
                cursor = max(cursor, pos + 120)
        return positions

    def _find_chunk_position(self, chunk: str, text: str, start: int) -> int | None:
        if not chunk:
            return None

        probes = [chunk[:220], chunk[:160], chunk[:100], chunk[:70]]
        for probe in probes:
            token = probe.strip()
            if len(token) < 24:
                continue
            idx = text.find(token, start)
            if idx != -1:
                return idx

        for probe in probes:
            token = probe.strip()
            if len(token) < 24:
                continue
            idx = text.find(token)
            if idx != -1:
                return idx

        return None

    def _event_for_position(
        self,
        position: int | None,
        events: Sequence[tuple[int, str]],
    ) -> str | None:
        if position is None or not events:
            return None
        active: str | None = None
        for offset, label in events:
            if offset > position:
                break
            active = label
        return active

    def _upsert_nodes(self, valid_nodes) -> None:
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.COLLECTION_NAME,
            enable_hybrid=True,
            sparse_doc_fn=self.sparse_doc_fn,
            sparse_query_fn=self.sparse_query_fn,
            batch_size=settings.UPSERT_BATCH_SIZE,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            valid_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
    
    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def _generate_summary_with_retry(self, prompt: str) -> str:
        """Generates summary with retry logic for API rate limits."""
        logger.info("Requesting LLM completion (with potential retry)...")
        response = await self.llm.acomplete(prompt)
        return str(response).strip()

    async def _generate_summary(self, markdown_text: str) -> str:
        summary_prompt = (
            "Summarize this document for retrieval support. "
            "Provide 4-6 concise bullets with core topics and key entities. "
            "Do not speculate.\n\n"
            f"Document excerpt:\n{markdown_text[:15000]}"
        )
        try:
            summary = await self._generate_summary_with_retry(summary_prompt)
            return summary[:2000]
        except Exception as exc:
            logger.warning("Summary generation failed, using fallback: %s", exc)
            return self._fallback_summary(markdown_text)

    def _fallback_summary(self, markdown_text: str) -> str:
        lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        top_lines = lines[:6]
        if not top_lines:
            return "Summary unavailable."
        bullets = "\n".join(f"- {line[:180]}" for line in top_lines)
        return f"Summary fallback from extracted content:\n{bullets}"

    def _count_chunks_for_doc(self, doc_id: str) -> int:
        try:
            result = self.qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            logger.info("Collection not ready for count (%s)", exc)
            return 0

    def _delete_points_for_filename(self, filename: str) -> int:
        try:
            existing = self.qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                ),
                exact=True,
            )
            delete_count = int(existing.count)
            if delete_count == 0:
                return 0

            self.qdrant_client.delete(
                collection_name=settings.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                ),
                wait=True,
            )
            logger.info("Deleted %s old chunks for filename=%s", delete_count, filename)
            return delete_count
        except Exception as exc:
            logger.info("Skipping replacement delete for filename=%s: %s", filename, exc)
            return 0
