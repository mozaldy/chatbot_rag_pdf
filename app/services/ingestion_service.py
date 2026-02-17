import asyncio
import base64
import os
import random
import shutil
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Sequence

from fastapi import UploadFile
import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PictureDescriptionApiOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling_core.types.doc import DocItemLabel, PictureClassificationLabel

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.core.llm_setup import get_fast_llm, get_embedding_model, get_sparse_embedding_functions
from app.services.rag_quality import (
    compute_chunk_id,
    compute_document_id,
    compute_point_uuid,
    normalize_markdown_text,
)

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Import common API errors if possible to be specific, or generic Exception for now
try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

logger = logging.getLogger(__name__)
_MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$")
_PAGE_PATTERN = re.compile(r"^page\s*(\d+)(?:\s*(?:of|/)\s*(\d+))?$", re.IGNORECASE)
_PAGE_RATIO_PATTERN = re.compile(r"^(\d+)\s*(?:of|/)\s*(\d+)$", re.IGNORECASE)
_PAGE_NUMERIC_PATTERN = re.compile(r"^\d{1,4}$")
_CHART_CLASSIFICATION_ALLOW = (
    PictureClassificationLabel.PIE_CHART,
    PictureClassificationLabel.BAR_CHART,
    PictureClassificationLabel.STACKED_BAR_CHART,
    PictureClassificationLabel.LINE_CHART,
    PictureClassificationLabel.FLOW_CHART,
    PictureClassificationLabel.SCATTER_CHART,
    PictureClassificationLabel.HEATMAP,
    PictureClassificationLabel.STRATIGRAPHIC_CHART,
)

class IngestionService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT,
        )
        self.llm = get_fast_llm()
        self.embed_model = get_embedding_model()
        self.sparse_doc_fn, self.sparse_query_fn = get_sparse_embedding_functions()
        self._table_visual_retry_queue: list[dict[str, Any]] = []

    async def process_pdf(
        self,
        file: UploadFile,
        include_markdown: bool = False,
    ):
        temp_path = self._save_upload_file(file)
        try:
            logger.info(f"Parsing PDF with Docling: {file.filename}")
            raw_markdown, page_segments, structure_map = await asyncio.to_thread(
                self._extract_docling_payload,
                temp_path,
            )
            normalized_markdown = normalize_markdown_text(raw_markdown)
            enriched_preview_markdown = self._build_preview_markdown(
                structure_map=structure_map,
                fallback_markdown=normalized_markdown,
            )

            if not normalized_markdown:
                raise ValueError("No readable content extracted from PDF.")

            doc_id = compute_document_id(file.filename, normalized_markdown)

            # Always clean up any existing chunks for this filename first.
            # This prevents ghost duplicates when normalization changes produce
            # a different doc_id for the same file.
            replaced_points = await asyncio.to_thread(
                self._delete_points_for_filename,
                file.filename,
            )
            if replaced_points > 0:
                logger.info(
                    "Cleaned up %s old chunks for %s before re-ingestion",
                    replaced_points,
                    file.filename,
                )

            logger.info("Generating global summary for %s", file.filename)
            global_summary = await self._generate_summary(normalized_markdown)

            valid_nodes = await asyncio.to_thread(
                self._build_nodes,
                normalized_markdown,
                raw_markdown,
                page_segments,
                doc_id,
                file.filename,
                structure_map,
            )

            chunk_kind_counts: dict[str, int] = {}
            for node in valid_nodes:
                metadata = node.metadata or {}
                chunk_kind = str(
                    metadata.get("chunk_kind")
                    or metadata.get("content_type")
                    or "text"
                )
                chunk_kind_counts[chunk_kind] = chunk_kind_counts.get(chunk_kind, 0) + 1

            table_visual_done_count = sum(
                1
                for s in structure_map
                if s.get("type") == "table" and s.get("table_visual_status") == "done"
            )
            table_visual_pending_count = sum(
                1
                for s in structure_map
                if s.get("type") == "table" and s.get("table_visual_status") == "pending"
            )
            table_visual_failed_count = sum(
                1
                for s in structure_map
                if s.get("type") == "table" and s.get("table_visual_status") == "failed"
            )
            table_visual_selected_count = sum(
                1
                for s in structure_map
                if s.get("type") == "table" and bool(s.get("table_visual_selected", False))
            )
            table_visual_skipped_count = sum(
                1
                for s in structure_map
                if s.get("type") == "table" and s.get("table_visual_status") == "skipped_by_router"
            )

            # --- Ingestion telemetry ---
            telemetry = {
                "table_count": sum(1 for s in structure_map if s["type"] == "table"),
                "table_visual_interpreted_count": table_visual_done_count,
                "table_visual_pending_count": table_visual_pending_count,
                "table_visual_failed_count": table_visual_failed_count,
                "table_visual_selected_count": table_visual_selected_count,
                "table_visual_skipped_count": table_visual_skipped_count,
                "table_retry_queue_size": len(self._table_visual_retry_queue),
                "figure_count": sum(1 for s in structure_map if s["type"] == "figure"),
                "text_count": sum(1 for s in structure_map if s["type"] == "text"),
                "table_parent_count": chunk_kind_counts.get("table_parent", 0),
                "table_anchor_count": chunk_kind_counts.get("table_anchor", 0),
                "total_chunks": len(valid_nodes),
                "empty_chunk_count": sum(
                    1 for n in valid_nodes if len(n.get_content().strip()) < 50
                ),
                "avg_chunk_len": round(
                    sum(len(n.get_content()) for n in valid_nodes)
                    / max(len(valid_nodes), 1)
                ),
                "ocr_enabled": settings.ENABLE_OCR,
                "chunking_schema_version": settings.CHUNKING_SCHEMA_VERSION,
            }
            logger.info("Ingestion telemetry for %s: %s", file.filename, telemetry)

            logger.info(f"Upserting {len(valid_nodes)} nodes to Qdrant...")
            await asyncio.to_thread(self._upsert_nodes, valid_nodes)

            return {
                "filename": file.filename,
                "status": "ingested",
                "chunks": len(valid_nodes),
                "global_summary": global_summary,
                "doc_id": doc_id,
                "replaced_points": replaced_points,
                "ingested_markdown": enriched_preview_markdown if include_markdown else None,
                "chunking_schema_version": settings.CHUNKING_SCHEMA_VERSION,
                "table_parent_count": chunk_kind_counts.get("table_parent", 0),
                "table_anchor_count": chunk_kind_counts.get("table_anchor", 0),
                "table_visual_pending_count": table_visual_pending_count,
                "table_visual_done_count": table_visual_done_count,
                "table_visual_failed_count": table_visual_failed_count,
                "table_visual_selected_count": table_visual_selected_count,
                "table_visual_skipped_count": table_visual_skipped_count,
                "chunk_diagnostics": self._build_chunk_diagnostics(valid_nodes),
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise e
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_upload_file(self, upload_file: UploadFile) -> str:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name

    def _extract_docling_payload(
        self, temp_path: str
    ) -> tuple[str, list[tuple[int, str]], list[dict]]:
        """Extract content from PDF with enhanced Docling configuration.

        Returns:
            raw_markdown: Full document markdown export.
            page_segments: Per-page text snippets for alignment.
            structure_map: List of structural elements with type, content, and page info.
        """
        pipeline_options = self._build_pdf_pipeline_options()

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(temp_path)

        doc = result.document

        raw_markdown = doc.export_to_markdown()
        page_segments = self._collect_page_segments(doc)
        structure_map = self._extract_structure_map(doc)
        self._enrich_table_visual_interpretations(doc, structure_map)
        self._strip_internal_structure_fields(structure_map)

        return raw_markdown, page_segments, structure_map

    def _build_pdf_pipeline_options(self) -> PdfPipelineOptions:
        """Create Docling pipeline options, including optional API picture description."""
        table_mode = (
            TableFormerMode.FAST
            if settings.TABLE_FORMER_MODE.lower() == "fast"
            else TableFormerMode.ACCURATE
        )
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = settings.ENABLE_OCR
        pipeline_options.do_table_structure = True
        pipeline_options.do_chart_extraction = settings.ENABLE_CHART_EXTRACTION
        pipeline_options.do_picture_description = settings.ENABLE_PICTURE_DESCRIPTION
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=table_mode,
        )
        if self._is_table_visual_interpretation_enabled():
            # Needed for TableItem.get_image(doc) in visual table enrichment.
            pipeline_options.generate_page_images = True
        self._configure_picture_description_options(pipeline_options)
        return pipeline_options

    def _configure_picture_description_options(
        self,
        pipeline_options: PdfPipelineOptions,
    ) -> None:
        provider = (settings.PICTURE_DESCRIPTION_PROVIDER or "local").strip().lower()
        if provider == "local":
            return

        if provider != "gemini_api":
            raise ValueError(
                "Unsupported PICTURE_DESCRIPTION_PROVIDER: "
                f"{settings.PICTURE_DESCRIPTION_PROVIDER!r}. "
                "Use 'local' or 'gemini_api'."
            )

        if settings.ENABLE_CHART_EXTRACTION:
            logger.warning(
                "ENABLE_CHART_EXTRACTION uses local Granite Vision in Docling. "
                "Gemini provider applies only to picture descriptions."
            )

        if not pipeline_options.do_picture_description:
            return

        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required when PICTURE_DESCRIPTION_PROVIDER=gemini_api."
            )
        if not settings.PICTURE_DESCRIPTION_API_URL:
            raise ValueError(
                "PICTURE_DESCRIPTION_API_URL is required when "
                "PICTURE_DESCRIPTION_PROVIDER=gemini_api."
            )

        params: dict[str, Any] = {}
        if settings.PICTURE_DESCRIPTION_MODEL:
            params["model"] = settings.PICTURE_DESCRIPTION_MODEL

        classification_allow = None
        if settings.PICTURE_DESCRIPTION_ONLY_CHARTS:
            # Filtering by chart labels requires picture classification metadata.
            pipeline_options.do_picture_classification = True
            classification_allow = list(_CHART_CLASSIFICATION_ALLOW)

        pipeline_options.enable_remote_services = True
        pipeline_options.picture_description_options = PictureDescriptionApiOptions(
            url=settings.PICTURE_DESCRIPTION_API_URL,
            headers={"Authorization": f"Bearer {settings.GOOGLE_API_KEY}"},
            params=params,
            timeout=settings.PICTURE_DESCRIPTION_TIMEOUT,
            concurrency=settings.PICTURE_DESCRIPTION_CONCURRENCY,
            prompt=settings.PICTURE_DESCRIPTION_PROMPT,
            classification_allow=classification_allow,
            provenance="gemini_api",
        )

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: int = 1) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(minimum, parsed)

    @staticmethod
    def _coerce_float(
        value: Any,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        if minimum is not None:
            parsed = max(minimum, parsed)
        if maximum is not None:
            parsed = min(maximum, parsed)
        return parsed

    def _is_table_visual_interpretation_enabled(self) -> bool:
        return self._coerce_bool(
            getattr(settings, "ENABLE_TABLE_VISUAL_INTERPRETATION", False),
            default=False,
        )

    def _table_visual_routing_mode(self) -> str:
        mode_raw = getattr(settings, "TABLE_VISUAL_ROUTING_MODE", "auto_signal")
        mode = str(mode_raw or "auto_signal").strip().lower()
        if mode in {"always", "always_on", "all"}:
            return "always_on"
        return "auto_signal"

    def _table_visual_signal_threshold(self) -> float:
        bias_raw = getattr(settings, "TABLE_VISUAL_DETECTOR_BIAS", "precision")
        bias = str(bias_raw or "precision").strip().lower()
        default_threshold = 0.75
        if bias == "balanced":
            default_threshold = 0.65
        elif bias == "recall":
            default_threshold = 0.55
        return self._coerce_float(
            getattr(settings, "TABLE_VISUAL_SIGNAL_THRESHOLD", default_threshold),
            default=default_threshold,
            minimum=0.0,
            maximum=1.0,
        )

    def _score_table_visual_need(self, entry: dict) -> tuple[float, list[str], bool]:
        score = 0.0
        reasons: list[str] = []

        empty_ratio = self._coerce_float(entry.get("table_empty_cell_ratio", 0.0), default=0.0, minimum=0.0, maximum=1.0)
        span_ratio = self._coerce_float(entry.get("table_span_cell_ratio", 0.0), default=0.0, minimum=0.0, maximum=1.0)
        num_rows = self._coerce_int(entry.get("table_num_rows", 0), default=0, minimum=0)
        num_cols = self._coerce_int(entry.get("table_num_cols", 0), default=0, minimum=0)
        lane_header_signal = bool(entry.get("table_lane_header_signal", False))
        strong_keyword_signal = bool(entry.get("table_strong_keyword_signal", False))

        if strong_keyword_signal:
            score += 0.85
            reasons.append("strong_visual_keyword")

        if lane_header_signal:
            score += 0.15
            reasons.append("lane_headers")

        if empty_ratio >= 0.60:
            score += 0.22
            reasons.append("high_empty_cell_ratio")
        elif empty_ratio >= 0.48:
            score += 0.14
            reasons.append("moderate_empty_cell_ratio")

        if num_cols >= 7:
            score += 0.08
            reasons.append("wide_table")

        if num_rows >= 8:
            score += 0.06
            reasons.append("long_table")

        if span_ratio >= 0.08:
            score += 0.12
            reasons.append("merged_cells")

        structural_flow_pattern = (
            lane_header_signal
            and num_cols >= 6
            and empty_ratio >= 0.50
            and (span_ratio >= 0.05 or num_rows >= 8)
        )
        if structural_flow_pattern:
            score = max(score, 0.82)
            reasons.append("structural_flow_pattern")

        score = min(1.0, score)
        strong_signal = strong_keyword_signal or structural_flow_pattern
        return score, reasons, strong_signal

    def _should_run_table_visual(self, entry: dict) -> tuple[bool, float, list[str]]:
        mode = self._table_visual_routing_mode()
        if mode == "always_on":
            return True, 1.0, ["routing_mode_always_on"]

        score, reasons, strong_signal = self._score_table_visual_need(entry)
        require_strong = self._coerce_bool(
            getattr(settings, "TABLE_VISUAL_REQUIRE_STRONG_SIGNALS", True),
            default=True,
        )
        threshold = self._table_visual_signal_threshold()

        selected = score >= threshold and (not require_strong or strong_signal)
        if selected and not reasons:
            reasons = ["score_threshold"]
        return selected, score, reasons

    def _enrich_table_visual_interpretations(
        self,
        doc: Any,
        structure_map: list[dict],
    ) -> None:
        """Append visual SOP-flow interpretation under each table using Gemini vision.

        Tables that fail due to transient API issues are marked pending and queued for retry.
        """
        if not self._is_table_visual_interpretation_enabled():
            return

        provider_raw = getattr(settings, "TABLE_VISUAL_INTERPRETATION_PROVIDER", "")
        provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else ""
        if provider != "gemini_api":
            logger.warning(
                "Skipping table visual interpretation: unsupported provider=%r",
                provider_raw,
            )
            return

        api_key = getattr(settings, "GOOGLE_API_KEY", None)
        api_url = getattr(settings, "TABLE_VISUAL_INTERPRETATION_API_URL", None)
        if not api_key or not api_url:
            logger.warning(
                "Skipping table visual interpretation: GOOGLE_API_KEY or "
                "TABLE_VISUAL_INTERPRETATION_API_URL is missing."
            )
            return

        table_entries = [entry for entry in structure_map if entry.get("type") == "table"]
        if not table_entries:
            return

        selected_entries: list[dict] = []
        skipped_count = 0
        for entry in table_entries:
            selected, score, reasons = self._should_run_table_visual(entry)
            entry["table_visual_score"] = round(score, 4)
            entry["table_visual_reasons"] = reasons
            entry["table_visual_selected"] = bool(selected)
            if not selected:
                entry["table_visual_status"] = "skipped_by_router"
                entry["table_visual_attempts"] = 0
                entry.pop("table_visual_last_error", None)
                skipped_count += 1
                continue
            selected_entries.append(entry)

        tasks: list[tuple[dict, Any, str | None]] = []
        for entry in selected_entries:
            image = self._extract_table_image(entry, doc)
            if image is None:
                entry["table_visual_status"] = "failed"
                entry["table_visual_attempts"] = 0
                entry["table_visual_last_error"] = "table_image_unavailable"
                continue
            image_path = self._persist_table_image_for_retry(entry, image)
            tasks.append((entry, image, image_path))

        if not tasks:
            logger.info(
                "Table visual routing skipped all tables or no table images were available: selected=%s skipped=%s total=%s.",
                len(selected_entries),
                skipped_count,
                len(table_entries),
            )
            return

        workers = self._coerce_int(
            getattr(settings, "TABLE_VISUAL_INTERPRETATION_CONCURRENCY", 1),
            default=1,
            minimum=1,
        )

        logger.info(
            "Running table visual interpretation for %s selected table(s) using %s worker(s) (skipped=%s total=%s).",
            len(tasks),
            workers,
            skipped_count,
            len(table_entries),
        )

        def _run(task: tuple[dict, Any, str | None]) -> tuple[dict, str | None, str, int, str | None, str | None]:
            entry, image, image_path = task
            interpretation, status, attempts, last_error = self._interpret_table_visual_with_retry(
                table_image=image,
                table_markdown=entry.get("content", ""),
            )
            return entry, interpretation, status, attempts, last_error, image_path

        enriched_count = 0
        pending_count = 0
        failed_count = 0

        if workers == 1:
            iterator = map(_run, tasks)
            for entry, interpretation, status, attempts, last_error, image_path in iterator:
                self._apply_table_visual_result(
                    entry=entry,
                    interpretation=interpretation,
                    status=status,
                    attempts=attempts,
                    last_error=last_error,
                    image_path=image_path,
                )
                if status == "done":
                    enriched_count += 1
                elif status == "pending":
                    pending_count += 1
                else:
                    failed_count += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for entry, interpretation, status, attempts, last_error, image_path in executor.map(_run, tasks):
                    self._apply_table_visual_result(
                        entry=entry,
                        interpretation=interpretation,
                        status=status,
                        attempts=attempts,
                        last_error=last_error,
                        image_path=image_path,
                    )
                    if status == "done":
                        enriched_count += 1
                    elif status == "pending":
                        pending_count += 1
                    else:
                        failed_count += 1

        logger.info(
            "Table visual interpretation completed: done=%s pending=%s failed=%s selected=%s skipped=%s total=%s.",
            enriched_count,
            pending_count,
            failed_count,
            len(tasks),
            skipped_count,
            len(table_entries),
        )

    def _apply_table_visual_result(
        self,
        entry: dict,
        interpretation: str | None,
        status: str,
        attempts: int,
        last_error: str | None,
        image_path: str | None,
    ) -> None:
        entry["table_visual_status"] = status
        entry["table_visual_attempts"] = attempts
        entry["table_visual_selected"] = True
        if last_error:
            entry["table_visual_last_error"] = last_error
        else:
            entry.pop("table_visual_last_error", None)

        if image_path:
            entry["table_visual_image_path"] = image_path

        if interpretation:
            entry["content"] = self._append_table_visual_interpretation(
                entry.get("content", ""),
                interpretation,
            )
            entry["table_has_chart_signals"] = True
            return

        if status == "pending":
            entry["content"] = self._append_table_visual_pending_marker(
                entry.get("content", ""),
                attempts=attempts,
                last_error=last_error,
            )
            self._enqueue_table_visual_retry(entry, image_path, last_error)

    def _interpret_table_visual_with_retry(
        self,
        table_image: Any,
        table_markdown: str,
    ) -> tuple[str | None, str, int, str | None]:
        retry_enabled = self._coerce_bool(
            getattr(settings, "TABLE_VISUAL_RETRY_ENABLED", True),
            default=True,
        )
        max_attempts = self._coerce_int(
            getattr(settings, "TABLE_VISUAL_MAX_RETRIES", 5),
            default=5,
            minimum=1,
        )
        if not retry_enabled:
            max_attempts = 1

        backoff_seconds = float(getattr(settings, "TABLE_VISUAL_BACKOFF_SECONDS", 2.0) or 2.0)
        backoff_multiplier = float(
            getattr(settings, "TABLE_VISUAL_BACKOFF_MULTIPLIER", 2.0) or 2.0
        )

        attempts = 0
        last_error: str | None = None
        had_retryable_failure = False

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            interpretation, retryable, error = self._interpret_table_visual_with_api_detailed(
                table_image=table_image,
                table_markdown=table_markdown,
            )
            if interpretation:
                return interpretation, "done", attempts, None

            if error:
                last_error = error
            had_retryable_failure = had_retryable_failure or retryable

            if not retry_enabled or not retryable or attempt >= max_attempts:
                break

            sleep_for = max(0.2, backoff_seconds * (backoff_multiplier ** (attempt - 1)))
            sleep_for += random.uniform(0.0, min(1.0, sleep_for * 0.2))
            time.sleep(sleep_for)

        status = "pending" if had_retryable_failure and retry_enabled else "failed"
        return None, status, attempts, last_error

    def _interpret_table_visual_with_api_detailed(
        self,
        table_image: Any,
        table_markdown: str,
    ) -> tuple[str | None, bool, str | None]:
        try:
            image_buffer = BytesIO()
            table_image.save(image_buffer, format="PNG")
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            error = f"serialize_error: {exc}"
            logger.warning("Failed to serialize table image for vision call: %s", exc)
            return None, False, error

        prompt = self._build_table_visual_prompt(table_markdown)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {"messages": messages}
        model = getattr(settings, "TABLE_VISUAL_INTERPRETATION_MODEL", None)
        if isinstance(model, str) and model.strip():
            payload["model"] = model.strip()

        timeout = float(getattr(settings, "TABLE_VISUAL_INTERPRETATION_TIMEOUT", 45.0))
        headers = {
            "Authorization": f"Bearer {settings.GOOGLE_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                str(settings.TABLE_VISUAL_INTERPRETATION_API_URL),
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except Exception as exc:
            error = f"request_error: {exc}"
            logger.warning("Table visual interpretation request failed: %s", exc)
            return None, True, error

        if not response.ok:
            status_code = int(response.status_code)
            response_excerpt = response.text[:500]
            logger.warning(
                "Table visual interpretation API returned %s: %s",
                status_code,
                response_excerpt,
            )
            retryable = status_code in {429, 500, 502, 503, 504}
            return None, retryable, f"http_{status_code}: {response_excerpt}"

        try:
            data = response.json()
        except Exception as exc:
            error = f"json_parse_error: {exc}"
            logger.warning("Failed to parse table interpretation response JSON: %s", exc)
            return None, False, error

        completion = self._extract_chat_completion_text(data)
        if not completion:
            return None, False, "empty_completion"

        max_chars = self._coerce_int(
            getattr(settings, "TABLE_VISUAL_INTERPRETATION_MAX_CHARS", 1600),
            default=1600,
            minimum=200,
        )
        normalized = re.sub(r"\n{3,}", "\n\n", completion).strip()
        if len(normalized) > max_chars:
            normalized = normalized[:max_chars].rstrip() + "..."
        return normalized, False, None

    def _interpret_table_visual_with_api(
        self,
        table_image: Any,
        table_markdown: str,
    ) -> str | None:
        text, _retryable, _error = self._interpret_table_visual_with_api_detailed(
            table_image=table_image,
            table_markdown=table_markdown,
        )
        return text

    def _append_table_visual_pending_marker(
        self,
        table_content: str,
        attempts: int,
        last_error: str | None,
    ) -> str:
        details = ""
        if last_error:
            details = f" Last error: {last_error[:280]}"
        marker = (
            "### Table Visual Interpretation\n"
            f"- Pending visual interpretation retry after {attempts} attempt(s).{details}"
        )
        if marker in table_content:
            return table_content
        return f"{table_content.rstrip()}\n\n{marker}\n"

    def _persist_table_image_for_retry(self, entry: dict, table_image: Any) -> str | None:
        try:
            retry_dir = Path("/tmp/rag_table_visual_retry")
            retry_dir.mkdir(parents=True, exist_ok=True)
            table_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(entry.get("table_id") or "table"))
            filename = f"{table_id}_{int(time.time() * 1000)}.png"
            image_path = retry_dir / filename
            table_image.save(image_path, format="PNG")
            return str(image_path)
        except Exception as exc:
            logger.warning("Failed to persist table image for retry queue: %s", exc)
            return None

    def _enqueue_table_visual_retry(
        self,
        entry: dict,
        image_path: str | None,
        last_error: str | None,
    ) -> None:
        if not image_path:
            return
        queue_item = {
            "table_id": entry.get("table_id"),
            "page_no": entry.get("page_no"),
            "image_path": image_path,
            "error": last_error,
            "queued_at": int(time.time()),
        }
        self._table_visual_retry_queue.append(queue_item)
        if len(self._table_visual_retry_queue) > 512:
            self._table_visual_retry_queue = self._table_visual_retry_queue[-512:]

    def _extract_table_image(self, entry: dict, doc: Any) -> Any | None:
        table_item = entry.get("_doc_item")
        if table_item is None:
            return None
        get_image = getattr(table_item, "get_image", None)
        if not callable(get_image):
            return None

        image = None
        try:
            image = get_image(doc)
        except TypeError:
            try:
                image = get_image(doc=doc)
            except Exception:
                return None
        except Exception:
            return None

        if image is None:
            return None

        mode = getattr(image, "mode", None)
        if mode and mode != "RGB":
            try:
                image = image.convert("RGB")
            except Exception:
                return None
        return image

    def _build_table_visual_prompt(self, table_markdown: str) -> str:
        base_prompt = getattr(
            settings,
            "TABLE_VISUAL_INTERPRETATION_PROMPT",
            "Describe visual process flow semantics not captured in table text.",
        )
        table_excerpt = (table_markdown or "").strip()[:2500]
        return (
            f"{base_prompt}\n\n"
            "Use this table markdown as context and focus on missing visual flow details:\n"
            f"```markdown\n{table_excerpt}\n```"
        )

    def _extract_chat_completion_text(self, payload: dict[str, Any]) -> str | None:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            joined = "\n".join(parts).strip()
            return joined or None
        return None

    def _append_table_visual_interpretation(
        self,
        table_content: str,
        interpretation: str,
    ) -> str:
        interpretation = interpretation.strip()
        if not interpretation:
            return table_content

        if not re.match(r"^(-|\*|\d+\.)\s+", interpretation):
            interpretation = f"- {interpretation}"

        block = f"### Table Visual Interpretation\n{interpretation}"
        if block in table_content:
            return table_content
        return f"{table_content.rstrip()}\n\n{block}\n"

    def _strip_internal_structure_fields(self, structure_map: list[dict]) -> None:
        for entry in structure_map:
            for key in list(entry.keys()):
                if key.startswith("_"):
                    entry.pop(key, None)

    def _build_preview_markdown(
        self,
        structure_map: Sequence[dict],
        fallback_markdown: str,
    ) -> str:
        """Build user-facing markdown preview from structure-aware content."""
        parts: list[str] = []
        for entry in structure_map:
            content = entry.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if not parts:
            return fallback_markdown
        return "\n\n".join(parts)

    def _extract_structure_map(self, doc: Any) -> list[dict]:
        """Walk Docling document items and classify each as table/figure/text."""
        structure_map: list[dict] = []
        table_counter = 0

        for item, _level in self._iterate_doc_items_with_labels(doc):
            label = getattr(item, "label", None)
            page_no = self._item_page_no(item)

            if label == DocItemLabel.TABLE:
                table_counter += 1
                table_id = f"table_{table_counter:04d}"
                try:
                    table_md = item.export_to_markdown(doc=doc)
                except Exception:
                    table_md = self._item_text(item) or ""
                caption = self._safe_caption(item, doc)
                content = f"**Table: {caption}**\n\n{table_md}" if caption else table_md
                profile = self._build_table_profile(content, caption or "", item)
                structure_map.append(
                    {
                        "type": "table",
                        "content": content,
                        "page_no": page_no,
                        "caption": caption or "",
                        "table_id": table_id,
                        "table_visual_status": "not_applicable",
                        "table_visual_attempts": 0,
                        "table_visual_last_error": "",
                        "table_visual_score": 0.0,
                        "table_visual_selected": False,
                        "table_visual_reasons": [],
                        **profile,
                        "_doc_item": item,
                    }
                )

            elif label in (DocItemLabel.PICTURE, DocItemLabel.CHART):
                caption = self._safe_caption(item, doc)
                description = self._safe_picture_description(item)
                parts = [part for part in (caption, description) if part]
                content = "\n\n".join(parts) if parts else f"[{label} on page {page_no or '?'}]"
                structure_map.append(
                    {"type": "figure", "content": content, "page_no": page_no, "caption": caption or ""}
                )

            elif label == DocItemLabel.CAPTION:
                # Captions are attached to the preceding table/figure; skip standalone.
                continue

            else:
                item_text = self._item_text(item)
                if item_text and len(item_text.strip()) > 2:
                    structure_map.append(
                        {"type": "text", "content": item_text, "page_no": page_no, "caption": ""}
                    )

        return structure_map

    def _has_chart_like_signals(self, text: str) -> bool:
        normalized = (text or "").lower()
        tokens = (
            "flow",
            "flowchart",
            "workflow",
            "decision",
            "diagram",
            "diagram alir",
            "chart",
            "arrow",
            "connector",
            "handoff",
            "swimlane",
            "panah",
        )
        return any(token in normalized for token in tokens)

    def _build_table_profile(
        self,
        table_content: str,
        caption: str,
        table_item: Any,
    ) -> dict[str, Any]:
        content = table_content or ""
        table_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]

        rows: list[list[str]] = []
        for line in table_lines:
            if re.match(r"^\|[\s:|-]+\|$", line):
                continue
            cells = self._split_markdown_row(line)
            if cells:
                rows.append(cells)

        header_cells = rows[0] if rows else []
        body_rows = rows[1:] if len(rows) > 1 else []
        num_cols_markdown = max((len(r) for r in rows), default=0)
        body_cell_total = max(len(body_rows) * max(num_cols_markdown, 1), 1)

        empty_cells = 0
        for row in body_rows:
            padded = row + [""] * max(0, num_cols_markdown - len(row))
            empty_cells += sum(1 for cell in padded if not (cell or "").strip())
        empty_ratio = empty_cells / float(body_cell_total) if body_rows else 0.0

        lane_header_signal = any(
            re.search(r"(pelaksana|admin|pimpinan|civitas|upt|mutu|waktu|output)", cell or "", re.IGNORECASE)
            for cell in header_cells
        )

        combined_text = f"{caption}\n{content}"
        keyword_signal = self._has_chart_like_signals(combined_text)
        strong_keyword_signal = bool(
            re.search(
                r"(flow\s*chart|workflow|diagram\s*alir|decision|panah|arrow|connector|swimlane)",
                combined_text,
                re.IGNORECASE,
            )
        )

        data = getattr(table_item, "data", None)
        data_num_rows = self._coerce_int(
            getattr(data, "num_rows", len(body_rows)),
            default=len(body_rows),
            minimum=0,
        )
        data_num_cols = self._coerce_int(
            getattr(data, "num_cols", num_cols_markdown),
            default=num_cols_markdown,
            minimum=0,
        )

        table_cells = list(getattr(data, "table_cells", []) or [])
        span_cells = 0
        for cell in table_cells:
            row_span = self._coerce_int(getattr(cell, "row_span", 1), default=1, minimum=1)
            col_span = self._coerce_int(getattr(cell, "col_span", 1), default=1, minimum=1)
            if row_span > 1 or col_span > 1:
                span_cells += 1
        span_ratio = (span_cells / float(len(table_cells))) if table_cells else 0.0

        structural_flow_pattern = (
            lane_header_signal
            and (data_num_cols or num_cols_markdown) >= 6
            and empty_ratio >= 0.50
            and (span_ratio >= 0.05 or (data_num_rows or len(body_rows)) >= 8)
        )

        has_chart_signals = keyword_signal or strong_keyword_signal or structural_flow_pattern

        return {
            "table_num_rows": int(data_num_rows or len(body_rows)),
            "table_num_cols": int(data_num_cols or num_cols_markdown),
            "table_empty_cell_ratio": round(float(empty_ratio), 4),
            "table_span_cell_ratio": round(float(span_ratio), 4),
            "table_lane_header_signal": bool(lane_header_signal),
            "table_keyword_signal": bool(keyword_signal),
            "table_strong_keyword_signal": bool(strong_keyword_signal),
            "table_has_chart_signals": bool(has_chart_signals),
        }

    def _iterate_doc_items_with_labels(self, document: Any) -> Iterable[tuple[Any, int]]:
        """Iterate document items, preferring the API that exposes labels."""
        iterate_items = getattr(document, "iterate_items", None)
        if callable(iterate_items):
            for kwargs in (
                {"with_groups": False, "traverse_pictures": True},
                {"with_groups": False, "traverse_pictures": False},
            ):
                try:
                    yield from iterate_items(**kwargs)
                    return
                except TypeError:
                    continue
                except Exception:
                    pass
            try:
                yield from iterate_items()
                return
            except Exception:
                pass
        # Fallback: wrap plain text items.
        for item in list(getattr(document, "texts", []) or []):
            yield item, 0

    def _safe_caption(self, item: Any, doc: Any) -> str:
        """Safely extract caption text from a Docling item."""
        caption_attr = getattr(item, "caption_text", None)
        if caption_attr is None:
            return ""
        try:
            if callable(caption_attr):
                result = caption_attr(doc)
            else:
                result = caption_attr
            return str(result).strip() if result else ""
        except Exception:
            return ""

    def _safe_picture_description(self, item: Any) -> str:
        """Best-effort extraction of picture description text from Docling metadata."""
        meta = getattr(item, "meta", None)
        if meta is not None:
            description_field = getattr(meta, "description", None)
            text = getattr(description_field, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

        annotations = getattr(item, "annotations", None) or []
        for annotation in annotations:
            text = getattr(annotation, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""

    # ------------------------------------------------------------------
    # Node building
    # ------------------------------------------------------------------

    def _build_nodes(
        self,
        normalized_markdown: str,
        raw_markdown: str,
        page_segments: Sequence[tuple[int, str]],
        doc_id: str,
        filename: str,
        structure_map: Sequence[dict] | None = None,
    ):
        if structure_map:
            nodes = self._build_structure_aware_nodes(
                structure_map, doc_id, filename
            )
        else:
            nodes = self._build_flat_nodes(
                normalized_markdown, doc_id, filename
            )

        if not nodes:
            raise ValueError("No chunks generated from parsed document.")

        # --- Page and section alignment (shared logic) ---
        section_events = self._extract_section_events(normalized_markdown)
        page_alignment_text, page_events = self._build_page_alignment(page_segments)
        if not page_alignment_text or not page_events:
            page_aware_markdown = normalize_markdown_text(
                raw_markdown, preserve_page_markers=True
            )
            page_alignment_text = page_aware_markdown
            page_events = self._extract_page_events(page_aware_markdown)

        section_positions = self._align_node_positions(nodes, normalized_markdown)
        page_positions = self._align_node_positions(nodes, page_alignment_text)

        valid_nodes = []
        for idx, node in enumerate(nodes):
            chunk_id = compute_chunk_id(doc_id, idx)
            node.id_ = compute_point_uuid(chunk_id)
            section_title = self._event_for_position(
                section_positions[idx], section_events
            )
            page_label = self._event_for_position(
                page_positions[idx], page_events
            )

            # Use page_no from structure_map if alignment missed.
            if not page_label and structure_map:
                sm_page = node.metadata.get("_struct_page_no")
                if sm_page is not None:
                    page_label = f"Page {sm_page}"

            node.metadata["filename"] = filename
            node.metadata["source_doc_id"] = doc_id
            node.metadata["chunk_index"] = idx
            node.metadata["chunk_id"] = chunk_id
            node.metadata["total_chunks"] = len(nodes)
            node.metadata["section_title"] = section_title or "Document"
            node.metadata.setdefault("schema_version", settings.CHUNKING_SCHEMA_VERSION)
            node.metadata.setdefault(
                "chunk_kind",
                str(node.metadata.get("content_type") or "text"),
            )
            node.metadata.setdefault("parent_id", None)
            node.metadata.setdefault("table_id", None)
            node.metadata.setdefault("table_visual_status", "not_applicable")
            node.metadata.setdefault("table_visual_attempts", 0)
            node.metadata.setdefault("table_visual_score", 0.0)
            node.metadata.setdefault("table_visual_selected", False)
            node.metadata.setdefault("table_visual_reasons", [])
            node.metadata.setdefault("table_has_chart_signals", False)
            if page_label:
                node.metadata["page_label"] = page_label
            # Clean up internal-only keys.
            node.metadata.pop("_struct_page_no", None)
            node.excluded_embed_metadata_keys.extend(
                [
                    "filename",
                    "source_doc_id",
                    "chunk_index",
                    "chunk_id",
                    "total_chunks",
                    "section_title",
                    "page_label",
                    "content_type",
                    "schema_version",
                    "chunk_kind",
                    "parent_id",
                    "table_id",
                    "table_visual_status",
                    "table_visual_attempts",
                    "table_visual_score",
                    "table_visual_selected",
                    "table_visual_reasons",
                    "table_visual_last_error",
                    "table_has_chart_signals",
                ]
            )
            valid_nodes.append(node)

        return valid_nodes

    def _build_structure_aware_nodes(
        self,
        structure_map: Sequence[dict],
        doc_id: str,
        filename: str,
    ) -> list:
        """Build nodes respecting structural boundaries.

        Tables become atomic parent chunks plus optional semantic anchors.
        Consecutive text blocks are merged and split with SentenceSplitter as before.
        """
        parser = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )
        all_nodes: list = []
        text_buffer: list[str] = []
        buffer_page: int | None = None
        table_counter = 0

        def flush_text_buffer():
            nonlocal text_buffer, buffer_page
            if not text_buffer:
                return
            merged = "\n\n".join(text_buffer)
            normalized = normalize_markdown_text(merged)
            if not normalized.strip():
                text_buffer = []
                buffer_page = None
                return
            docs = [
                Document(
                    text=normalized,
                    metadata={
                        "filename": filename,
                        "source_doc_id": doc_id,
                        "content_type": "text",
                        "chunk_kind": "text",
                        "schema_version": settings.CHUNKING_SCHEMA_VERSION,
                        "_struct_page_no": buffer_page,
                    },
                )
            ]
            nodes = parser.get_nodes_from_documents(docs)
            for node in nodes:
                node.metadata["content_type"] = "text"
                node.metadata["chunk_kind"] = "text"
                node.metadata["schema_version"] = settings.CHUNKING_SCHEMA_VERSION
                node.metadata["_struct_page_no"] = buffer_page
            all_nodes.extend(nodes)
            text_buffer = []
            buffer_page = None

        for entry in structure_map:
            entry_type = entry["type"]
            content = entry.get("content", "")
            page_no = entry.get("page_no")

            if entry_type == "text":
                text_buffer.append(content)
                if buffer_page is None and page_no is not None:
                    buffer_page = page_no
                continue

            # Flush any accumulated text before a structural element.
            flush_text_buffer()

            if entry_type == "table":
                table_counter += 1
                table_id = str(entry.get("table_id") or f"{doc_id}:table:{table_counter:04d}")
                table_nodes = self._chunk_table(
                    table_content=content,
                    doc_id=doc_id,
                    filename=filename,
                    page_no=page_no,
                    caption=entry.get("caption", ""),
                    table_id=table_id,
                    visual_status=str(entry.get("table_visual_status") or "not_applicable"),
                    visual_attempts=int(entry.get("table_visual_attempts") or 0),
                    visual_last_error=(entry.get("table_visual_last_error") or "") or None,
                    visual_score=self._coerce_float(entry.get("table_visual_score", 0.0), default=0.0, minimum=0.0, maximum=1.0),
                    visual_selected=bool(entry.get("table_visual_selected", False)),
                    visual_reasons=list(entry.get("table_visual_reasons") or []),
                    table_has_chart_signals=bool(entry.get("table_has_chart_signals", False)),
                )
                all_nodes.extend(table_nodes)

            elif entry_type == "figure":
                if content.strip():
                    from llama_index.core.schema import TextNode as TNode
                    fig_node = TNode(
                        text=content,
                        metadata={
                            "filename": filename,
                            "source_doc_id": doc_id,
                            "content_type": "figure",
                            "chunk_kind": "figure",
                            "schema_version": settings.CHUNKING_SCHEMA_VERSION,
                            "_struct_page_no": page_no,
                        },
                    )
                    all_nodes.append(fig_node)

        # Flush remaining text.
        flush_text_buffer()
        return all_nodes

    def _chunk_table(
        self,
        table_content: str,
        doc_id: str,
        filename: str,
        page_no: int | None,
        caption: str,
        table_id: str,
        visual_status: str,
        visual_attempts: int,
        visual_last_error: str | None,
        visual_score: float,
        visual_selected: bool,
        visual_reasons: list[str],
        table_has_chart_signals: bool,
    ) -> list:
        """Build one table parent node and optional semantic anchor nodes."""
        from llama_index.core.schema import TextNode as TNode

        content = (table_content or "").strip()
        if caption and not content.startswith("**Table:"):
            content = f"**Table: {caption}**\n\n{content}"

        base_meta: dict[str, Any] = {
            "filename": filename,
            "source_doc_id": doc_id,
            "content_type": "table",
            "schema_version": settings.CHUNKING_SCHEMA_VERSION,
            "table_id": table_id,
            "table_visual_status": visual_status,
            "table_visual_attempts": max(0, int(visual_attempts)),
            "table_visual_score": round(float(visual_score), 4),
            "table_visual_selected": bool(visual_selected),
            "table_visual_reasons": list(visual_reasons or []),
            "table_has_chart_signals": bool(table_has_chart_signals),
            "_struct_page_no": page_no,
        }
        if visual_last_error:
            base_meta["table_visual_last_error"] = visual_last_error

        parent_meta = dict(base_meta)
        parent_meta["chunk_kind"] = "table_parent"
        parent_meta["parent_id"] = None
        parent_node = TNode(text=content, metadata=parent_meta)

        nodes: list[TNode] = [parent_node]

        if not self._coerce_bool(getattr(settings, "ENABLE_TABLE_ANCHORS", True), default=True):
            return nodes

        anchor_limit = self._coerce_int(
            getattr(settings, "TABLE_ANCHOR_MAX_PER_TABLE", 6),
            default=6,
            minimum=1,
        )
        anchor_texts = self._build_table_anchor_texts(content, caption, anchor_limit)
        for anchor_index, anchor_text in enumerate(anchor_texts, start=1):
            anchor_meta = dict(base_meta)
            anchor_meta["chunk_kind"] = "table_anchor"
            anchor_meta["parent_id"] = table_id
            anchor_meta["table_anchor_index"] = anchor_index
            nodes.append(TNode(text=anchor_text, metadata=anchor_meta))

        return nodes

    def _build_table_anchor_texts(
        self,
        table_content: str,
        caption: str,
        max_anchors: int,
    ) -> list[str]:
        lines = [line.strip() for line in (table_content or "").splitlines() if line.strip()]
        table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]

        header_cells: list[str] = []
        body_rows: list[list[str]] = []

        for line in table_lines:
            if re.match(r"^\|[\s:|-]+\|$", line):
                continue
            cells = self._split_markdown_row(line)
            if not cells:
                continue
            if not header_cells:
                header_cells = cells
                continue
            body_rows.append(cells)

        anchors: list[str] = []

        if caption:
            anchors.append(f"Table caption: {caption}")

        if header_cells:
            anchors.append("Table headers: " + ", ".join(header_cells[:12]))

        role_terms = [
            cell
            for cell in header_cells
            if re.search(r"(pelaksana|admin|pimpinan|civitas|upt|mutu|output|waktu)", cell, re.IGNORECASE)
        ]
        if role_terms:
            anchors.append("Roles and lanes: " + ", ".join(role_terms[:10]))

        if body_rows:
            activity_values = [row[1] for row in body_rows if len(row) > 1 and row[1]]
            if activity_values:
                anchors.append("Main activities: " + " | ".join(activity_values[:8]))

            duration_values: list[str] = []
            output_values: list[str] = []
            duration_idx = -1
            output_idx = -1
            for idx, col_name in enumerate(header_cells):
                lowered = col_name.lower()
                if duration_idx == -1 and re.search(r"(waktu|durasi|lama|time)", lowered):
                    duration_idx = idx
                if output_idx == -1 and re.search(r"output", lowered):
                    output_idx = idx

            if duration_idx >= 0:
                for row in body_rows:
                    if duration_idx < len(row) and row[duration_idx]:
                        duration_values.append(row[duration_idx])
            if output_idx >= 0:
                for row in body_rows:
                    if output_idx < len(row) and row[output_idx]:
                        output_values.append(row[output_idx])

            if duration_values:
                anchors.append("Timing constraints: " + ", ".join(duration_values[:10]))
            if output_values:
                anchors.append("Expected outputs: " + ", ".join(output_values[:10]))

        visual_bullets = self._extract_table_visual_bullets(table_content)
        if visual_bullets:
            anchors.append("Visual flow cues: " + " | ".join(visual_bullets[:4]))

        deduped: list[str] = []
        seen: set[str] = set()
        for anchor in anchors:
            normalized = re.sub(r"\s+", " ", anchor).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= max_anchors:
                break
        return deduped

    @staticmethod
    def _split_markdown_row(line: str) -> list[str]:
        if not (line.startswith("|") and line.endswith("|")):
            return []
        raw_cells = line.strip("|").split("|")
        return [cell.strip() for cell in raw_cells]

    @staticmethod
    def _extract_table_visual_bullets(table_content: str) -> list[str]:
        if "### Table Visual Interpretation" not in table_content:
            return []
        section = table_content.split("### Table Visual Interpretation", 1)[1]
        bullets: list[str] = []
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped:
                if bullets:
                    break
                continue
            if stripped.startswith("-") or re.match(r"^\d+\.\s+", stripped):
                bullets.append(stripped.lstrip("-0123456789. ").strip())
        return [b for b in bullets if b]

    def _build_flat_nodes(
        self,
        normalized_markdown: str,
        doc_id: str,
        filename: str,
    ) -> list:
        """Legacy flat chunking fallback."""
        parser = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )
        documents = [
            Document(
                text=normalized_markdown,
                id_=doc_id,
                metadata={
                    "filename": filename,
                    "source_doc_id": doc_id,
                    "content_type": "text",
                },
            )
        ]
        nodes = parser.get_nodes_from_documents(documents)
        for node in nodes:
            node.metadata["content_type"] = "text"
        return nodes

    def _collect_page_segments(self, document: Any) -> list[tuple[int, str]]:
        segments: list[tuple[int, str]] = []

        for item in self._iterate_doc_items(document):
            page_no = self._item_page_no(item)
            if page_no is None:
                continue
            text = self._item_text(item)
            if not text:
                continue
            normalized = self._normalize_for_alignment(text)
            if len(normalized) < 24:
                continue
            segments.append((page_no, normalized))

        if segments:
            return segments

        # Fallback for older Docling versions: export page-by-page markdown.
        pages = sorted((getattr(document, "pages", {}) or {}).keys())
        for page_no in pages:
            try:
                page_md = document.export_to_markdown(page_no=page_no)
            except Exception:
                continue
            normalized = self._normalize_for_alignment(page_md)
            if len(normalized) >= 24:
                segments.append((int(page_no), normalized))
        return segments

    def _iterate_doc_items(self, document: Any) -> Iterable[Any]:
        iterate_items = getattr(document, "iterate_items", None)
        if callable(iterate_items):
            for kwargs in (
                {"with_groups": False, "traverse_pictures": True},
                {"with_groups": False, "traverse_pictures": False},
            ):
                try:
                    for item, _level in iterate_items(**kwargs):
                        yield item
                    return
                except TypeError:
                    continue
                except Exception:
                    pass
            try:
                for item, _level in iterate_items():
                    yield item
                return
            except Exception:
                pass

        for item in list(getattr(document, "texts", []) or []):
            yield item

    def _item_text(self, item: Any) -> str | None:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        return None

    def _item_page_no(self, item: Any) -> int | None:
        prov_list = getattr(item, "prov", None)
        if not prov_list:
            return None
        for prov in prov_list:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                return page_no
        return None

    def _normalize_for_alignment(self, text: str) -> str:
        normalized = normalize_markdown_text(text, preserve_page_markers=True)
        return re.sub(r"\s+", " ", normalized).strip()

    def _build_page_alignment(
        self,
        page_segments: Sequence[tuple[int, str]],
    ) -> tuple[str, list[tuple[int, str]]]:
        parts: list[str] = []
        events: list[tuple[int, str]] = []
        cursor = 0

        for page_no, segment_text in page_segments:
            if not segment_text:
                continue
            events.append((cursor, f"Page {page_no}"))
            parts.append(segment_text)
            cursor += len(segment_text) + 2

        return "\n\n".join(parts), events

    def _extract_section_events(self, text: str) -> list[tuple[int, str]]:
        events: list[tuple[int, str]] = [(0, "Document")]
        cursor = 0
        for line in text.splitlines(keepends=True):
            match = _MARKDOWN_HEADING_PATTERN.match(line.strip())
            if match:
                title = match.group(1).strip()
                if title:
                    events.append((cursor, title[:200]))
            cursor += len(line)
        return events

    def _extract_page_events(self, text: str) -> list[tuple[int, str]]:
        events: list[tuple[int, str]] = []
        cursor = 0
        for line in text.splitlines(keepends=True):
            page_label = self._canonical_page_label(line.strip())
            if page_label:
                events.append((cursor, page_label))
            cursor += len(line)
        return events

    def _canonical_page_label(self, value: str) -> str | None:
        if not value:
            return None

        match = _PAGE_PATTERN.fullmatch(value)
        if match:
            return f"Page {int(match.group(1))}"

        match = _PAGE_RATIO_PATTERN.fullmatch(value)
        if match:
            return f"Page {int(match.group(1))}"

        if _PAGE_NUMERIC_PATTERN.fullmatch(value):
            number = int(value)
            if 1 <= number <= 5000:
                return f"Page {number}"
        return None

    def _align_node_positions(
        self,
        nodes: Sequence,
        text: str,
    ) -> list[int | None]:
        positions: list[int | None] = []
        cursor = 0
        for node in nodes:
            content = (node.get_content() or "").strip()
            pos = self._find_chunk_position(content, text, cursor)
            positions.append(pos)
            if pos is not None:
                cursor = max(cursor, pos + 120)
        return positions

    def _find_chunk_position(self, chunk: str, text: str, start: int) -> int | None:
        if not chunk:
            return None

        probes = [chunk[:220], chunk[:160], chunk[:100], chunk[:70]]
        for probe in probes:
            token = probe.strip()
            if len(token) < 24:
                continue
            idx = text.find(token, start)
            if idx != -1:
                return idx

        for probe in probes:
            token = probe.strip()
            if len(token) < 24:
                continue
            idx = text.find(token)
            if idx != -1:
                return idx

        return None

    def _event_for_position(
        self,
        position: int | None,
        events: Sequence[tuple[int, str]],
    ) -> str | None:
        if position is None or not events:
            return None
        active: str | None = None
        for offset, label in events:
            if offset > position:
                break
            active = label
        return active

    def _build_chunk_diagnostics(self, nodes: Sequence[Any]) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for node in nodes:
            metadata = node.metadata or {}
            diagnostics.append(
                {
                    "chunk_id": metadata.get("chunk_id", node.node_id),
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_kind": metadata.get("chunk_kind") or metadata.get("content_type") or "text",
                    "content_type": metadata.get("content_type", "text"),
                    "table_id": metadata.get("table_id"),
                    "parent_id": metadata.get("parent_id"),
                    "page_label": metadata.get("page_label"),
                    "section_title": metadata.get("section_title"),
                    "char_len": len((node.get_content() or "").strip()),
                    "table_visual_status": metadata.get("table_visual_status"),
                    "table_visual_score": metadata.get("table_visual_score"),
                    "table_visual_selected": metadata.get("table_visual_selected"),
                    "table_visual_reasons": metadata.get("table_visual_reasons"),
                    "schema_version": metadata.get("schema_version"),
                }
            )
        return diagnostics

    def _upsert_nodes(self, valid_nodes) -> None:
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.COLLECTION_NAME,
            enable_hybrid=True,
            sparse_doc_fn=self.sparse_doc_fn,
            sparse_query_fn=self.sparse_query_fn,
            batch_size=settings.UPSERT_BATCH_SIZE,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            valid_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
    
    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def _generate_summary_with_retry(self, prompt: str) -> str:
        """Generates summary with retry logic for API rate limits."""
        logger.info("Requesting LLM completion (with potential retry)...")
        response = await self.llm.acomplete(prompt)
        return str(response).strip()

    async def _generate_summary(self, markdown_text: str) -> str:
        summary_prompt = (
            "Summarize this document for retrieval support. "
            "Provide 4-6 concise bullets with core topics and key entities. "
            "Do not speculate.\n\n"
            f"Document excerpt:\n{markdown_text[:15000]}"
        )
        try:
            summary = await self._generate_summary_with_retry(summary_prompt)
            return summary[:2000]
        except Exception as exc:
            logger.warning("Summary generation failed, using fallback: %s", exc)
            return self._fallback_summary(markdown_text)

    def _fallback_summary(self, markdown_text: str) -> str:
        lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        top_lines = lines[:6]
        if not top_lines:
            return "Summary unavailable."
        bullets = "\n".join(f"- {line[:180]}" for line in top_lines)
        return f"Summary fallback from extracted content:\n{bullets}"

    def _count_chunks_for_doc(self, doc_id: str) -> int:
        try:
            result = self.qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="source_doc_id", match=MatchValue(value=doc_id))]
                ),
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            logger.info("Collection not ready for count (%s)", exc)
            return 0

    def _delete_points_for_filename(self, filename: str) -> int:
        try:
            existing = self.qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                ),
                exact=True,
            )
            delete_count = int(existing.count)
            if delete_count == 0:
                return 0

            self.qdrant_client.delete(
                collection_name=settings.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                ),
                wait=True,
            )
            logger.info("Deleted %s old chunks for filename=%s", delete_count, filename)
            return delete_count
        except Exception as exc:
            logger.info("Skipping replacement delete for filename=%s: %s", filename, exc)
            return 0
