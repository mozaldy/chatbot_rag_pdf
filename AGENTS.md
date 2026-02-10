# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: FastAPI entrypoint; mounts routes from `app/api/endpoints.py`.
- `app/api/`: HTTP handlers (`/ingest`, `/chat`, `/reset`).
- `app/services/`: core ingestion and retrieval logic (`ingestion_service.py`, `rag_quality.py`).
- `app/core/`: runtime config and model setup (`config.py`, `llm_setup.py`).
- `app/db/`: Qdrant integration utilities.
- `app/models/`: Pydantic request/response schemas.
- `tests/`: unit tests (`test_*.py`).
- Root scripts: `streamlit_app.py` (UI), `evaluate_rag.py` (quality evaluation), `test_rag.py` (manual API smoke script).

## Build, Test, and Development Commands
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker-compose up -d
python3 main.py
streamlit run streamlit_app.py
python3 evaluate_rag.py
python3 -m unittest discover -s tests -p "test_*.py"
```
- Use `docker-compose up -d` to start Qdrant on `localhost:6333`.
- Use `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` when you want explicit server flags.

## Coding Style & Naming Conventions
- Follow existing Python conventions: 4-space indentation, PEP 8 spacing, and type hints on public functions.
- Use `snake_case` for files, functions, and variables; use `PascalCase` for classes and Pydantic models.
- Keep API handlers thin; move business logic into `app/services/`.
- Group imports as standard library, third-party, and local modules.

## Testing Guidelines
- Primary framework is `unittest` (see `tests/test_rag_quality.py`).
- Name files `test_*.py` and methods `test_<behavior>`.
- Prefer deterministic unit tests for text normalization, ranking, and source assembly logic.
- Use `test_rag.py` only as a manual integration smoke check against a running API.

## Commit & Pull Request Guidelines
- Use Conventional Commit prefixes as seen in history (`feat:`, `fix:`).
- Keep commits focused and include tests/docs when behavior changes.
- PRs should include purpose, key changes, test command output, and example API/UI behavior for user-facing changes.

## Security & Configuration Tips
- Copy `.env.example` to `.env`; never commit secrets.
- Validate provider settings (`OLLAMA_BASE_URL`, API keys) before ingestion/chat runs.
- `DELETE /api/reset` wipes the Qdrant collection; use it only in local development/testing.
