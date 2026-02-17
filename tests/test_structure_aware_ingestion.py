"""Tests for structure-aware extraction and layout-aware chunking.

Covers:
- Table chunks are kept atomic (never split mid-row)
- Oversized tables split at row boundaries with header preserved
- Figure chunks carry caption text
- content_type metadata is set correctly
- Normalization preserves markdown table pipe chars
- Boilerplate filter does not drop table rows
"""

import re
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.rag_quality import normalize_markdown_text
from app.services.ingestion_service import IngestionService


class PipelineOptionConfigTests(unittest.TestCase):
    """Validate Docling pipeline option wiring for picture description providers."""

    def setUp(self):
        self.service = IngestionService.__new__(IngestionService)

    def _set_common_pdf_settings(self, mock_settings):
        mock_settings.TABLE_FORMER_MODE = "accurate"
        mock_settings.ENABLE_OCR = True
        mock_settings.ENABLE_CHART_EXTRACTION = False
        mock_settings.ENABLE_PICTURE_DESCRIPTION = True
        mock_settings.PICTURE_DESCRIPTION_PROVIDER = "local"
        mock_settings.PICTURE_DESCRIPTION_MODEL = "unused"
        mock_settings.PICTURE_DESCRIPTION_API_URL = "https://example.com/v1/chat/completions"
        mock_settings.PICTURE_DESCRIPTION_TIMEOUT = 20.0
        mock_settings.PICTURE_DESCRIPTION_CONCURRENCY = 1
        mock_settings.PICTURE_DESCRIPTION_ONLY_CHARTS = False
        mock_settings.PICTURE_DESCRIPTION_PROMPT = "unused"
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.ENABLE_TABLE_VISUAL_INTERPRETATION = False
        mock_settings.TABLE_VISUAL_INTERPRETATION_PROVIDER = "gemini_api"
        mock_settings.TABLE_VISUAL_INTERPRETATION_MODEL = "gemini-2.5-flash-lite"
        mock_settings.TABLE_VISUAL_INTERPRETATION_API_URL = "https://example.com/v1/chat/completions"
        mock_settings.TABLE_VISUAL_INTERPRETATION_TIMEOUT = 45.0
        mock_settings.TABLE_VISUAL_INTERPRETATION_CONCURRENCY = 2
        mock_settings.TABLE_VISUAL_INTERPRETATION_MAX_CHARS = 1600
        mock_settings.TABLE_VISUAL_INTERPRETATION_PROMPT = "visual prompt"
        mock_settings.TABLE_VISUAL_ROUTING_MODE = "auto_signal"
        mock_settings.TABLE_VISUAL_DETECTOR_BIAS = "precision"
        mock_settings.TABLE_VISUAL_SIGNAL_THRESHOLD = 0.75
        mock_settings.TABLE_VISUAL_REQUIRE_STRONG_SIGNALS = True

    @patch("app.services.ingestion_service.settings")
    def test_gemini_picture_description_configured_via_api_options(self, mock_settings):
        self._set_common_pdf_settings(mock_settings)
        mock_settings.PICTURE_DESCRIPTION_PROVIDER = "gemini_api"
        mock_settings.PICTURE_DESCRIPTION_MODEL = "gemini-2.0-flash"
        mock_settings.PICTURE_DESCRIPTION_API_URL = (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        mock_settings.PICTURE_DESCRIPTION_TIMEOUT = 45.0
        mock_settings.PICTURE_DESCRIPTION_CONCURRENCY = 2
        mock_settings.PICTURE_DESCRIPTION_ONLY_CHARTS = True
        mock_settings.PICTURE_DESCRIPTION_PROMPT = "Describe chart details."
        mock_settings.GOOGLE_API_KEY = "test-google-key"

        options = self.service._build_pdf_pipeline_options()

        self.assertTrue(options.enable_remote_services)
        self.assertTrue(options.do_picture_description)
        self.assertTrue(options.do_picture_classification)
        self.assertEqual(options.picture_description_options.kind, "api")
        self.assertEqual(
            str(options.picture_description_options.url),
            mock_settings.PICTURE_DESCRIPTION_API_URL,
        )
        self.assertEqual(
            options.picture_description_options.headers.get("Authorization"),
            "Bearer test-google-key",
        )
        self.assertEqual(
            options.picture_description_options.params.get("model"),
            "gemini-2.0-flash",
        )
        self.assertEqual(options.picture_description_options.timeout, 45.0)
        self.assertEqual(options.picture_description_options.concurrency, 2)
        self.assertIsNotNone(options.picture_description_options.classification_allow)
        self.assertGreater(len(options.picture_description_options.classification_allow), 0)

    @patch("app.services.ingestion_service.settings")
    def test_local_picture_description_leaves_remote_services_disabled(self, mock_settings):
        self._set_common_pdf_settings(mock_settings)

        options = self.service._build_pdf_pipeline_options()

        self.assertFalse(options.enable_remote_services)
        self.assertEqual(options.picture_description_options.kind, "vlm")

    @patch("app.services.ingestion_service.settings")
    def test_gemini_provider_requires_google_api_key(self, mock_settings):
        self._set_common_pdf_settings(mock_settings)
        mock_settings.PICTURE_DESCRIPTION_PROVIDER = "gemini_api"
        mock_settings.PICTURE_DESCRIPTION_MODEL = "gemini-2.0-flash"
        mock_settings.PICTURE_DESCRIPTION_API_URL = "https://example.com/v1/chat/completions"
        mock_settings.PICTURE_DESCRIPTION_ONLY_CHARTS = True
        mock_settings.PICTURE_DESCRIPTION_PROMPT = "Describe chart details."
        mock_settings.GOOGLE_API_KEY = None

        with self.assertRaises(ValueError):
            self.service._build_pdf_pipeline_options()

    @patch("app.services.ingestion_service.settings")
    def test_unsupported_picture_description_provider_raises(self, mock_settings):
        self._set_common_pdf_settings(mock_settings)
        mock_settings.PICTURE_DESCRIPTION_PROVIDER = "unknown_provider"
        mock_settings.GOOGLE_API_KEY = "test-google-key"

        with self.assertRaises(ValueError):
            self.service._build_pdf_pipeline_options()

    @patch("app.services.ingestion_service.settings")
    def test_table_visual_interpretation_enables_page_images(self, mock_settings):
        self._set_common_pdf_settings(mock_settings)
        mock_settings.ENABLE_TABLE_VISUAL_INTERPRETATION = True

        options = self.service._build_pdf_pipeline_options()

        self.assertTrue(options.generate_page_images)

    @patch("app.services.ingestion_service.settings")
    def test_table_visual_interpretation_disabled_keeps_page_images_default(self, mock_settings):
        self._set_common_pdf_settings(mock_settings)
        mock_settings.ENABLE_TABLE_VISUAL_INTERPRETATION = False

        options = self.service._build_pdf_pipeline_options()

        self.assertFalse(options.generate_page_images)


class TableVisualInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.service = IngestionService.__new__(IngestionService)

    @patch("app.services.ingestion_service.settings")
    def test_enrich_table_visual_interpretation_appends_block(self, mock_settings):
        mock_settings.ENABLE_TABLE_VISUAL_INTERPRETATION = True
        mock_settings.TABLE_VISUAL_INTERPRETATION_PROVIDER = "gemini_api"
        mock_settings.GOOGLE_API_KEY = "test-google-key"
        mock_settings.TABLE_VISUAL_INTERPRETATION_API_URL = "https://example.com/v1/chat/completions"
        mock_settings.TABLE_VISUAL_INTERPRETATION_CONCURRENCY = 1
        mock_settings.TABLE_VISUAL_INTERPRETATION_MODEL = "gemini-test"
        mock_settings.TABLE_VISUAL_INTERPRETATION_TIMEOUT = 30.0
        mock_settings.TABLE_VISUAL_INTERPRETATION_MAX_CHARS = 1200
        mock_settings.TABLE_VISUAL_INTERPRETATION_PROMPT = "Describe SOP flow."
        mock_settings.TABLE_VISUAL_ROUTING_MODE = "always_on"
        mock_settings.TABLE_VISUAL_DETECTOR_BIAS = "precision"
        mock_settings.TABLE_VISUAL_SIGNAL_THRESHOLD = 0.75
        mock_settings.TABLE_VISUAL_REQUIRE_STRONG_SIGNALS = True

        fake_image = MagicMock()
        fake_image.mode = "RGB"
        table_item = MagicMock()
        table_item.get_image = MagicMock(return_value=fake_image)
        structure_map = [
            {
                "type": "table",
                "content": "| A | B |\n|---|---|\n| 1 | 2 |",
                "_doc_item": table_item,
            }
        ]

        with patch.object(
            self.service,
            "_interpret_table_visual_with_retry",
            return_value=(
                "- Flow starts at Civitas and moves to Admin Jaringan.",
                "done",
                1,
                None,
            ),
        ):
            self.service._enrich_table_visual_interpretations(MagicMock(), structure_map)

        self.assertIn("### Table Visual Interpretation", structure_map[0]["content"])
        self.assertIn("Flow starts", structure_map[0]["content"])
        self.assertEqual(structure_map[0].get("table_visual_status"), "done")

    @patch("app.services.ingestion_service.settings")
    def test_enrich_table_visual_interpretation_skips_non_visual_tables(self, mock_settings):
        mock_settings.ENABLE_TABLE_VISUAL_INTERPRETATION = True
        mock_settings.TABLE_VISUAL_INTERPRETATION_PROVIDER = "gemini_api"
        mock_settings.GOOGLE_API_KEY = "test-google-key"
        mock_settings.TABLE_VISUAL_INTERPRETATION_API_URL = "https://example.com/v1/chat/completions"
        mock_settings.TABLE_VISUAL_INTERPRETATION_CONCURRENCY = 1
        mock_settings.TABLE_VISUAL_ROUTING_MODE = "auto_signal"
        mock_settings.TABLE_VISUAL_DETECTOR_BIAS = "precision"
        mock_settings.TABLE_VISUAL_SIGNAL_THRESHOLD = 0.75
        mock_settings.TABLE_VISUAL_REQUIRE_STRONG_SIGNALS = True

        table_item = MagicMock()
        structure_map = [
            {
                "type": "table",
                "content": "| No | Kegiatan | Waktu |\n|---|---|---|\n| 1 | Isi formulir | 5 menit |",
                "_doc_item": table_item,
                "table_num_rows": 2,
                "table_num_cols": 3,
                "table_empty_cell_ratio": 0.0,
                "table_span_cell_ratio": 0.0,
                "table_lane_header_signal": False,
                "table_strong_keyword_signal": False,
            }
        ]

        with patch.object(
            self.service,
            "_interpret_table_visual_with_retry",
            return_value=(
                "- Should not be used.",
                "done",
                1,
                None,
            ),
        ) as mock_interpret:
            self.service._enrich_table_visual_interpretations(MagicMock(), structure_map)

        mock_interpret.assert_not_called()
        self.assertEqual(structure_map[0].get("table_visual_status"), "skipped_by_router")
        self.assertFalse(structure_map[0].get("table_visual_selected"))
        self.assertEqual(structure_map[0].get("table_visual_score"), 0.0)

    def test_append_table_visual_interpretation_is_idempotent(self):
        table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
        once = self.service._append_table_visual_interpretation(
            table_md, "Flow goes from A to B."
        )
        twice = self.service._append_table_visual_interpretation(
            once, "Flow goes from A to B."
        )
        self.assertEqual(once, twice)

    def test_build_preview_markdown_uses_structure_content(self):
        structure_map = [
            {"type": "text", "content": "Intro"},
            {"type": "table", "content": "| A | B |\n|---|---|\n|1|2|\n\n### Table Visual Interpretation\n- A to B"},
        ]
        preview = self.service._build_preview_markdown(
            structure_map=structure_map,
            fallback_markdown="fallback",
        )
        self.assertIn("Intro", preview)
        self.assertIn("Table Visual Interpretation", preview)

    def test_build_preview_markdown_falls_back_when_empty(self):
        preview = self.service._build_preview_markdown(
            structure_map=[{"type": "text", "content": ""}],
            fallback_markdown="fallback",
        )
        self.assertEqual(preview, "fallback")


class NormalizeTablePreservationTests(unittest.TestCase):
    """Verify normalize_markdown_text preserves table markdown."""

    def test_table_pipe_chars_preserved(self):
        """Pipe-delimited table rows should keep their spacing."""
        table = (
            "| Name    | Age | City      |\n"
            "|---------|-----|-----------|\n"
            "| Alice   | 30  | New York  |\n"
            "| Bob     | 25  | London    |"
        )
        result = normalize_markdown_text(table)
        # All pipe rows should remain intact.
        for line in result.split("\n"):
            stripped = line.strip()
            if stripped:
                self.assertTrue(
                    stripped.startswith("|"),
                    f"Table row lost pipe prefix: {stripped!r}",
                )

    def test_non_table_whitespace_still_collapsed(self):
        """Regular text with extra spaces should still be collapsed."""
        text = "Hello    world,   this   is   a   test."
        result = normalize_markdown_text(text)
        self.assertNotIn("    ", result)

    def test_table_rows_not_dropped_by_boilerplate_filter(self):
        """Repeated table header rows should NOT be treated as boilerplate."""
        # Simulate a header row repeated many times (like across page breaks).
        lines = []
        for _ in range(10):
            lines.append("| Name | Value |")
            lines.append("|------|-------|")
            lines.append("| foo  | bar   |")
            lines.append("")
        text = "\n".join(lines)
        result = normalize_markdown_text(text)
        # The header row should survive.
        self.assertIn("| Name | Value |", result)

    def test_mixed_table_and_text(self):
        """Tables and regular text in the same document."""
        text = (
            "Some intro  text   with   spaces.\n\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| 1     | 2     |\n\n"
            "More   text   here."
        )
        result = normalize_markdown_text(text)
        # Table rows preserved.
        self.assertIn("| Col A | Col B |", result)
        self.assertIn("| 1     | 2     |", result)
        # Non-table whitespace collapsed.
        self.assertNotIn("  text   with   spaces", result)


class StructureMapExtractionTests(unittest.TestCase):
    """Test _extract_structure_map with mocked Docling doc items."""

    def setUp(self):
        self.service = IngestionService.__new__(IngestionService)

    def _make_item(self, label, text="", caption_text=None, page_no=None):
        item = MagicMock()
        item.label = label
        item.text = text
        item.export_to_markdown = MagicMock(return_value=text)
        if caption_text is not None:
            item.caption_text = MagicMock(return_value=caption_text)
        else:
            item.caption_text = None
        # prov for page number
        if page_no is not None:
            prov_item = MagicMock()
            prov_item.page_no = page_no
            item.prov = [prov_item]
        else:
            item.prov = []
        return item

    def test_table_item_classified_correctly(self):
        from docling_core.types.doc import DocItemLabel

        table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
        item = self._make_item(DocItemLabel.TABLE, table_md, caption_text="Sales Data", page_no=3)

        doc = MagicMock()
        doc.iterate_items = MagicMock(return_value=[(item, 0)])

        result = self.service._extract_structure_map(doc)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "table")
        self.assertIn("Sales Data", result[0]["content"])
        self.assertIn("| A | B |", result[0]["content"])
        self.assertEqual(result[0]["page_no"], 3)

    def test_figure_item_carries_caption(self):
        from docling_core.types.doc import DocItemLabel

        item = self._make_item(
            DocItemLabel.PICTURE, "", caption_text="Figure 1: Architecture", page_no=5
        )

        doc = MagicMock()
        doc.iterate_items = MagicMock(return_value=[(item, 0)])

        result = self.service._extract_structure_map(doc)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "figure")
        self.assertIn("Architecture", result[0]["content"])

    def test_text_item_classified_as_text(self):
        from docling_core.types.doc import DocItemLabel

        item = self._make_item(DocItemLabel.PARAGRAPH, "Regular paragraph text.")
        doc = MagicMock()
        doc.iterate_items = MagicMock(return_value=[(item, 0)])

        result = self.service._extract_structure_map(doc)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "text")

    def test_caption_items_skipped_standalone(self):
        from docling_core.types.doc import DocItemLabel

        item = self._make_item(DocItemLabel.CAPTION, "Table 1 caption only")
        doc = MagicMock()
        doc.iterate_items = MagicMock(return_value=[(item, 0)])

        result = self.service._extract_structure_map(doc)
        self.assertEqual(len(result), 0, "Standalone CAPTION items should be skipped")

    def test_iterate_items_prefers_traverse_pictures_true(self):
        calls = []
        item = MagicMock()

        def fake_iterate_items(**kwargs):
            calls.append(kwargs)
            return [(item, 0)]

        doc = MagicMock()
        doc.iterate_items = fake_iterate_items

        result = list(self.service._iterate_doc_items_with_labels(doc))
        self.assertEqual(len(result), 1)
        self.assertEqual(
            calls[0],
            {"with_groups": False, "traverse_pictures": True},
        )

    def test_iterate_doc_items_prefers_traverse_pictures_true(self):
        calls = []
        item = MagicMock()

        def fake_iterate_items(**kwargs):
            calls.append(kwargs)
            return [(item, 0)]

        doc = MagicMock()
        doc.iterate_items = fake_iterate_items

        result = list(self.service._iterate_doc_items(doc))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], item)
        self.assertEqual(
            calls[0],
            {"with_groups": False, "traverse_pictures": True},
        )


class LayoutAwareChunkingTests(unittest.TestCase):
    """Test structure-aware node building with atomic table parents and anchors."""

    def setUp(self):
        self.service = IngestionService.__new__(IngestionService)

    def test_table_kept_as_single_chunk(self):
        """Small tables should become one atomic chunk."""
        structure_map = [
            {"type": "text", "content": "Introduction paragraph.", "page_no": 1, "caption": ""},
            {
                "type": "table",
                "content": "| A | B |\n|---|---|\n| 1 | 2 |",
                "page_no": 2,
                "caption": "",
            },
            {"type": "text", "content": "Conclusion paragraph.", "page_no": 3, "caption": ""},
        ]
        nodes = self.service._build_structure_aware_nodes(
            structure_map, doc_id="test_doc", filename="test.pdf"
        )
        table_parent_nodes = [
            n for n in nodes if n.metadata.get("chunk_kind") == "table_parent"
        ]
        table_anchor_nodes = [
            n for n in nodes if n.metadata.get("chunk_kind") == "table_anchor"
        ]
        self.assertEqual(len(table_parent_nodes), 1)
        self.assertGreaterEqual(len(table_anchor_nodes), 1)
        self.assertIn("| A | B |", table_parent_nodes[0].get_content())
        self.assertIn("|---|---|", table_parent_nodes[0].get_content())

    def test_table_not_split_mid_row(self):
        """Ensure table chunks always contain complete rows."""
        structure_map = [
            {
                "type": "table",
                "content": "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
                "page_no": 1,
                "caption": "",
            },
        ]
        nodes = self.service._build_structure_aware_nodes(
            structure_map, doc_id="test_doc", filename="test.pdf"
        )
        for node in nodes:
            content = node.get_content()
            # Every line that starts with | should be a complete row (end with |).
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("|"):
                    self.assertTrue(
                        stripped.endswith("|"),
                        f"Row split mid-cell: {stripped!r}",
                    )

    @patch("app.services.ingestion_service.settings")
    def test_oversized_table_keeps_atomic_parent_with_anchors(self, mock_settings):
        """Large tables should stay whole as one parent node with separate anchor nodes."""
        mock_settings.CHUNK_SIZE = 768
        mock_settings.CHUNK_OVERLAP = 200
        mock_settings.CHUNKING_SCHEMA_VERSION = 2
        mock_settings.ENABLE_TABLE_ANCHORS = True
        mock_settings.TABLE_ANCHOR_MAX_PER_TABLE = 6

        header = "| No. | Kegiatan | Waktu | Output |\n|-----|----------|-------|--------|"
        rows = [
            f"| {i} | Aktivitas panjang nomor {i} | {i} menit | Output {i} |"
            for i in range(1, 40)
        ]
        table_content = header + "\n" + "\n".join(rows)

        nodes = self.service._build_structure_aware_nodes(
            [
                {
                    "type": "table",
                    "content": table_content,
                    "page_no": 1,
                    "caption": "",
                    "table_id": "table_0001",
                    "table_visual_status": "done",
                    "table_visual_attempts": 1,
                    "table_has_chart_signals": True,
                }
            ],
            doc_id="doc_1",
            filename="test.pdf",
        )

        parent_nodes = [n for n in nodes if n.metadata.get("chunk_kind") == "table_parent"]
        anchor_nodes = [n for n in nodes if n.metadata.get("chunk_kind") == "table_anchor"]

        self.assertEqual(len(parent_nodes), 1)
        self.assertGreaterEqual(len(anchor_nodes), 1)
        self.assertLessEqual(len(anchor_nodes), 6)
        parent_text = parent_nodes[0].get_content()
        self.assertIn("| No. | Kegiatan | Waktu | Output |", parent_text)
        self.assertIn("Aktivitas panjang nomor 39", parent_text)

    def test_content_type_metadata_set(self):
        """Each node should have content_type metadata."""
        structure_map = [
            {"type": "text", "content": "Some text content.", "page_no": 1, "caption": ""},
            {
                "type": "table",
                "content": "| X |\n|---|\n| 1 |",
                "page_no": 1,
                "caption": "",
            },
            {"type": "figure", "content": "Figure 1: Chart", "page_no": 2, "caption": "Chart"},
        ]
        nodes = self.service._build_structure_aware_nodes(
            structure_map, doc_id="test_doc", filename="test.pdf"
        )
        content_types = {n.metadata.get("content_type") for n in nodes}
        self.assertIn("text", content_types)
        self.assertIn("table", content_types)
        self.assertIn("figure", content_types)

    def test_figure_node_has_caption_text(self):
        """Figure nodes should contain the caption as their text content."""
        structure_map = [
            {"type": "figure", "content": "Figure 3: Revenue Growth Q1-Q4", "page_no": 5, "caption": "Revenue Growth Q1-Q4"},
        ]
        nodes = self.service._build_structure_aware_nodes(
            structure_map, doc_id="test_doc", filename="test.pdf"
        )
        self.assertEqual(len(nodes), 1)
        self.assertIn("Revenue Growth", nodes[0].get_content())


if __name__ == "__main__":
    unittest.main()
