import os
import shutil
import logging
from typing import List
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from docling.document_converter import DocumentConverter

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
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
            # 1. Parsing with Docling
            logger.info(f"Parsing PDF with Docling: {file.filename}")
            converter = DocumentConverter()
            result = converter.convert(temp_path)
            # Export to markdown to preserve table structures
            md_content = result.document.export_to_markdown()

            # 2. Generate Global Summary
            logger.info("Generating Global Summary...")
            # We take a sensible prefix of the document to avoid context window overflow on the generation step if strict.
            # Gemma 3 4b context window checks? Assuming it handles a reasonable amount.
            # Truncate for safety to ~10k chars for summary generation prompt if needed, 
            # or rely on Ollama/LlamaIndex truncating. Let's send the first 8000 words roughly or just full logic.
            # For robustness, let's take the first 15000 characters for the summary context.
            context_for_summary = md_content[:15000] 
            summary_prompt = (
                "You are a helpful AI assistant. Read the following document excerpt and provide a "
                "comprehensive 'Global Summary' of the main topics, purpose, and key findings. "
                "Keep it concise but informative (around 200 words).\n\n"
                f"Document Content:\n{context_for_summary}"
            )
            
            global_summary = await self._generate_summary_with_retry(summary_prompt)
            logger.info(f"Global Summary generated: {global_summary[:50]}...")

            # 3. Chunking & Enrichment
            logger.info("Chunking and enriching...")
            # Use MarkdownNodeParser since we have structure
            parser = MarkdownNodeParser()
            documents = [Document(text=md_content, metadata={"filename": file.filename})]
            nodes = parser.get_nodes_from_documents(documents)

            # Prepend Global Summary to each node's text or metadata for retrieval context
            valid_nodes = []
            for node in nodes:
                # We prepend to the text to ensure the embedding captures this context effectively
                # "Parent Document Context: {summary} \n\n Section Content: {node_text}"
                original_text = node.get_content()
                enriched_text = (
                    f"Global Document Context: {global_summary}\n\n---\n\n{original_text}"
                )
                node.set_content(enriched_text)
                
                # Keep original metadata and add summary
                node.metadata["global_summary"] = global_summary
                node.metadata["filename"] = file.filename
                
                # Exclude embedding logic from here, LlamaIndex handles it in ingestion
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
