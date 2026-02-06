from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App
    API_TITLE: str = "Local RAG API"
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "pdf_rag"

    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512  # Tokens per chunk (larger = more context)
    CHUNK_OVERLAP: int = 128  # Overlap between chunks for continuity
    
    # PDF Processing
    ENABLE_OCR: bool = True  # Set to True if you have scanned docs or mixed content
    
    # LLM Configuration (Main/Inference)
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "gemma3:4b"
    
    # LLM Configuration (Fast/Enrichment)
    FAST_LLM_PROVIDER: str = "ollama" 
    FAST_LLM_MODEL: str = "gemma3:4b"
    
    # Provider Specifics
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: float = 120.0
    
    # Cloud Providers (Optional)
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
