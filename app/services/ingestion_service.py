import os
import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

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
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = settings.ENABLE_OCR
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(temp_path)
            raw_markdown = result.document.export_to_markdown()
            normalized_markdown = normalize_markdown_text(raw_markdown)

            if not normalized_markdown:
                raise ValueError("No readable content extracted from PDF.")

            doc_id = compute_document_id(file.filename, normalized_markdown)
            existing_chunks = self._count_chunks_for_doc(doc_id)
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

            replaced_points = self._delete_points_for_filename(file.filename)
            logger.info(
                "Prepared ingestion for %s (doc_id=%s, replaced_points=%s)",
                file.filename,
                doc_id,
                replaced_points,
            )

            logger.info("Generating global summary for %s", file.filename)
            global_summary = await self._generate_summary(normalized_markdown)

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
                    "filename": file.filename,
                    "doc_id": doc_id,
                }
            )]
            nodes = parser.get_nodes_from_documents(documents)
            if not nodes:
                raise ValueError("No chunks generated from parsed document.")

            valid_nodes = []
            for idx, node in enumerate(nodes):
                chunk_id = compute_chunk_id(doc_id, idx)
                node.id_ = compute_point_uuid(chunk_id)
                node.metadata["filename"] = file.filename
                node.metadata["doc_id"] = doc_id
                node.metadata["chunk_index"] = idx
                node.metadata["chunk_id"] = chunk_id
                node.metadata["total_chunks"] = len(nodes)
                node.excluded_embed_metadata_keys.extend(
                    ["filename", "doc_id", "chunk_index", "chunk_id", "total_chunks"]
                )
                valid_nodes.append(node)

            logger.info(f"Upserting {len(valid_nodes)} nodes to Qdrant...")
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
