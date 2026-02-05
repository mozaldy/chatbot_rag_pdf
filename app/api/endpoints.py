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

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Quick retrieval logic implementation
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
        
        # 2. Query Engine
        # We need to manually set the LLM here to ensure it uses our Singleton
        llm = get_llm()
        
        # Using hybrid mode in retriever
        retriever = index.as_retriever(
            vector_store_query_mode="hybrid", 
            similarity_top_k=5,
            sparse_top_k=5
        )
        
        # Synthesis
        from llama_index.core.query_engine import RetrieverQueryEngine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm
        )
        
        response = await query_engine.aquery(request.messages)
        
        return {
            "response": str(response),
            "sources": [node.node.metadata.get("filename", "unknown") for node in response.source_nodes]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
