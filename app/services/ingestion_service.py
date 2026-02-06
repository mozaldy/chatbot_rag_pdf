import os
import shutil
import logging
from typing import List
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.llm_setup import get_llm, get_fast_llm, get_embedding_model, get_sparse_embedding_functions

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Import common API errors if possible to be specific, or generic Exception for now
try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        # We use the sync QdrantClient for LlamaIndex indexing flow usually,
        # or we can use the async one if properly configured.
        # LlamaIndex's QdrantVectorStore defaults to sync client under the hood for some ops.
        # Let's use the standard client for the vector store.
        self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        # Use Fast LLM for ingestion/enrichment tasks
        self.llm = get_fast_llm()
        self.embed_model = get_embedding_model()
        self.sparse_doc_fn, self.sparse_query_fn = get_sparse_embedding_functions()

    async def process_pdf(self, file: UploadFile):
        temp_path = self._save_upload_file(file)
        try:
            # 1. Parsing with Docling (OCR disabled for text-based PDFs)
            logger.info(f"Parsing PDF with Docling: {file.filename}")
            
            # Configure pipeline to skip OCR for text-based PDFs
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = settings.ENABLE_OCR
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(temp_path)
            # Export to markdown to preserve table structures
            md_content = result.document.export_to_markdown()
            
            logger.info(f"Extracted {len(md_content)} characters from PDF")

            # 2. Generate Global Summary
            logger.info("Generating Global Summary...")
            # Take first 15000 characters for summary context
            context_for_summary = md_content[:15000] 
            summary_prompt = (
                "You are a helpful AI assistant. Read the following document excerpt and provide a "
                "comprehensive 'Global Summary' of the main topics, purpose, and key findings. "
                "List ALL major features and capabilities mentioned. "
                "Keep it concise but informative (around 200 words).\n\n"
                f"Document Content:\n{context_for_summary}"
            )
            
            global_summary = await self._generate_summary_with_retry(summary_prompt)
            logger.info(f"Global Summary generated: {global_summary[:100]}...")

            # 3. Chunking & Enrichment
            logger.info("Chunking and enriching...")
            
            # Use SentenceSplitter for better semantic chunking
            parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
            )
            
            # CRITICAL FIX: Do NOT include global_summary in the Document metadata passed to the splitter.
            # The splitter validates that (text + metadata) < chunk_size.
            # Since global_summary is huge (~1000 chars), it causes a ValueError.
            # We will inject it manually into the nodes AFTER splitting.
            documents = [Document(
                text=md_content, 
                metadata={
                    "filename": file.filename,
                    # "global_summary": global_summary  <-- REMOVED from here to fix ValueError
                }
            )]
            nodes = parser.get_nodes_from_documents(documents)

            # Store metadata without prepending to text
            # This keeps embeddings focused on actual content
            valid_nodes = []
            for idx, node in enumerate(nodes):
                # Keep original text - don't dilute with summary
                # Metadata is accessible during retrieval
                node.metadata["filename"] = file.filename
                
                # INJECT SUMMARY HERE
                node.metadata["global_summary"] = global_summary
                
                node.metadata["chunk_index"] = idx
                node.metadata["total_chunks"] = len(nodes)
                
                valid_nodes.append(node)

            # 4. Storage (Dense + Sparse/Hybrid)
            logger.info(f"Upserting {len(valid_nodes)} nodes to Qdrant...")
            
            # Setup Hybrid Qdrant Vector Store
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.COLLECTION_NAME,
                enable_hybrid=True, # Critical for bge-m3 sparse generation
                sparse_doc_fn=self.sparse_doc_fn,
                sparse_query_fn=self.sparse_query_fn,
                batch_size=20 # Conservative batch size for 6GB VRAM
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create Index (triggers embedding generation)
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
                "global_summary": global_summary
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
