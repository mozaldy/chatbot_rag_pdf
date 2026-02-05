from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    messages: str # Simplification for single query
    
class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    
class IngestionResponse(BaseModel):
    filename: str
    status: str
    chunks: int
    global_summary: str
