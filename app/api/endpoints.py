from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient

from app.services.ingestion_service import IngestionService
from app.core.config import settings
from app.core.llm_setup import get_llm, get_embedding_model, get_sparse_embedding_functions
from app.models.schemas import ChatRequest, ChatResponse, IngestionResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    service = IngestionService()
    result = await service.process_pdf(file)
    return result

@router.delete("/reset", response_model=dict)
async def reset_database():
    """Wipes the vector database collection to start fresh."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        client.delete_collection(collection_name=settings.COLLECTION_NAME)
        return {"status": "success", "message": f"Collection '{settings.COLLECTION_NAME}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Enhanced retrieval logic with better parameters
    try:
        # 1. Setup Retriever
        client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
        # Singleton models
        embed_model = get_embedding_model()
        sparse_doc_fn, sparse_query_fn = get_sparse_embedding_functions()
        
        vector_store = QdrantVectorStore(
            aclient=client,
            collection_name=settings.COLLECTION_NAME,
            enable_hybrid=True,
            sparse_doc_fn=sparse_doc_fn,
            sparse_query_fn=sparse_query_fn
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        # 2. Query Engine with improved retrieval
        llm = get_llm()
        
        # Retrieve MORE candidates for better coverage
        # Hybrid search combines dense + sparse for better recall
        retriever = index.as_retriever(
            vector_store_query_mode="hybrid", 
            similarity_top_k=10,  # Increased from 5
            sparse_top_k=10,  # Increased from 5
        )
        
        # Add similarity threshold filtering
        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.response_synthesizers import ResponseMode
        
        # Filter out low-relevance results
        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.5)
        ]
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm,
            node_postprocessors=node_postprocessors,
            response_mode=ResponseMode.COMPACT,  # Better synthesis
        )
        
        response = await query_engine.aquery(request.messages)
        
        # Enhanced source tracking
        sources = []
        for node in response.source_nodes:
            source_info = {
                "filename": node.node.metadata.get("filename", "unknown"),
                "chunk_index": node.node.metadata.get("chunk_index", "?"),
                "score": round(node.score, 3) if hasattr(node, 'score') else None
            }
            sources.append(f"{source_info['filename']} (chunk {source_info['chunk_index']}, score: {source_info['score']})")
        
        return {
            "response": str(response),
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
