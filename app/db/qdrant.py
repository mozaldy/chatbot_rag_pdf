from qdrant_client import AsyncQdrantClient
from .config import settings

def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT
    )
