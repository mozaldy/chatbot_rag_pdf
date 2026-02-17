from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
    """Structured source information for interactive viewing."""

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
    content_type: Optional[str] = None  # "text", "table", "figure"
    chunk_kind: Optional[str] = None  # "table_parent", "table_anchor", etc.
    parent_id: Optional[str] = None
    table_id: Optional[str] = None
    schema_version: Optional[int] = None
    table_visual_status: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[SourceInfo]


class IngestionResponse(BaseModel):
    filename: str
    status: str
    chunks: int = 0
    global_summary: str = ""
    doc_id: Optional[str] = None
    replaced_points: int = 0
    ingested_markdown: Optional[str] = None
    chunking_schema_version: int = 2
    table_parent_count: int = 0
    table_anchor_count: int = 0
    table_visual_pending_count: int = 0
    table_visual_done_count: int = 0
    table_visual_failed_count: int = 0
    table_visual_selected_count: int = 0
    table_visual_skipped_count: int = 0
    chunk_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
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
    table_parent_chunks: int = 0
    table_anchor_chunks: int = 0
    chunk_kind_counts: Dict[str, int] = Field(default_factory=dict)
    schema_versions: List[int] = Field(default_factory=list)


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
