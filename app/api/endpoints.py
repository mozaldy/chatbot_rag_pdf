from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from qdrant_client import QdrantClient

from app.services.ingestion_service import IngestionService
from app.services.document_lifecycle_service import DocumentLifecycleService
from app.services.retrieval_service import RetrievalService
from app.core.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    IngestionBatchResponse,
    IngestionResponse,
)

router = APIRouter()


@router.post("/ingest", response_model=IngestionBatchResponse)
async def ingest_document(files: list[UploadFile] = File(..., alias="file")):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for ingestion.")

    service = IngestionService()
    results: list[IngestionResponse] = []

    for upload in files:
        filename = upload.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            results.append(
                IngestionResponse(
                    filename=filename,
                    status="failed",
                    error="Only PDF files are supported.",
                )
            )
            continue

        try:
            result = await service.process_pdf(upload)
            results.append(IngestionResponse(**result))
        except Exception as exc:
            results.append(
                IngestionResponse(
                    filename=filename,
                    status="failed",
                    error=str(exc),
                )
            )

    succeeded = sum(1 for item in results if item.status != "failed")
    failed = len(results) - succeeded
    return {
        "results": results,
        "total_files": len(results),
        "succeeded": succeeded,
        "failed": failed,
    }

@router.delete("/reset", response_model=dict)
async def reset_database():
    """Wipes the vector database collection to start fresh."""
    try:
        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT,
        )
        if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
            return {
                "status": "success",
                "message": f"Collection '{settings.COLLECTION_NAME}' is already absent.",
            }
        client.delete_collection(collection_name=settings.COLLECTION_NAME)
        return {"status": "success", "message": f"Collection '{settings.COLLECTION_NAME}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        service = RetrievalService()
        return await service.answer_query(request.messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    max_points: int | None = Query(default=None, ge=1, le=100000),
):
    try:
        service = DocumentLifecycleService()
        return service.list_documents(max_points=max_points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/by-filename", response_model=DocumentDeleteResponse)
async def delete_document_by_filename(
    filename: str = Query(..., min_length=1),
):
    try:
        service = DocumentLifecycleService()
        result = service.delete_by_filename(filename=filename)
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}", response_model=DocumentSummary)
async def get_document(doc_id: str):
    try:
        service = DocumentLifecycleService()
        doc = service.get_document(doc_id=doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    try:
        service = DocumentLifecycleService()
        result = service.delete_by_doc_id(doc_id=doc_id)
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
