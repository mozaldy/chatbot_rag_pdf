import logging
from fastapi import FastAPI
from app.api.endpoints import router
from app.core.llm_setup import init_settings

# Configure Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Local RAG API")

@app.on_event("startup")
def on_startup():
    # Initialize LlamaIndex Settings (Singleton loading)
    init_settings()

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
