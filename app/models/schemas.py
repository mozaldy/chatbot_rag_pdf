from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("message content must not be empty")
        return content

class ChatRequest(BaseModel):
    messages: str | List[ChatMessage]

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: str | List[ChatMessage]) -> str | List[ChatMessage]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("messages string must not be empty")
            return value
        if not value:
            raise ValueError("messages list must not be empty")
        return value

class SourceInfo(BaseModel):
    """Structured source information for interactive viewing"""
    id: str
    filename: str
    doc_id: str
    chunk_id: str
    chunk_index: int | str  # Can be int or "?" if unknown
    page_label: Optional[str] = None
    section_title: Optional[str] = None
    score: Optional[float]
    text: str  # The actual markdown chunk content
    node_id: str  # Qdrant node ID
    
class ChatResponse(BaseModel):
    response: str
    sources: List[SourceInfo]  # Changed from List[str] to structured objects
    
class IngestionResponse(BaseModel):
    filename: str
    status: str
    chunks: int = 0
    global_summary: str = ""
    doc_id: Optional[str] = None
    replaced_points: int = 0
    error: Optional[str] = None


class IngestionBatchResponse(BaseModel):
    results: List[IngestionResponse]
    total_files: int
    succeeded: int
    failed: int


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    max_chunk_index: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
    total_documents: int
    total_chunks: int
    scanned_points: int
    truncated: bool = False


class DocumentDeleteResponse(BaseModel):
    status: str
    message: str
    deleted_chunks: int
    doc_id: Optional[str] = None
    filename: Optional[str] = None
