from enum import Enum
from pydantic import BaseModel, Field

class RouteIntention(str, Enum):
    SEARCH = "SEARCH"
    SUMMARIZATION = "SUMMARIZATION"
    CHIT_CHAT = "CHIT_CHAT"
    AMBIGUOUS = "AMBIGUOUS"

class RouteResult(BaseModel):
    intention: RouteIntention = Field(..., description="The classified intention of the user query.")
    rewritten_query: str = Field(..., description="The query optimized for vector search/retrieval. If intention is not SEARCH, this can be the original query or empty.")
    reasoning: str = Field(..., description="Brief explanation of why this intention was chosen.")
    clarification_question: str | None = Field(None, description="A question to ask the user if the intention is AMBIGUOUS.")
