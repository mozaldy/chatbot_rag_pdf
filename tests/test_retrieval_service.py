import unittest

from app.services.retrieval_service import RetrievalService


class RetrievalServiceCitationTests(unittest.TestCase):
    def test_extract_citation_ids_reads_unique_source_ids(self) -> None:
        answer = "Feature A is supported [S1]. Another claim [S2] and repeat [S1]."
        citations = RetrievalService._extract_citation_ids(answer)
        self.assertEqual(citations, {"S1", "S2"})

    def test_citation_ids_are_valid_only_when_subset_of_sources(self) -> None:
        sources = [{"id": "S1"}, {"id": "S2"}, {"id": "S3"}]
        self.assertTrue(RetrievalService._citation_ids_are_valid({"S1", "S3"}, sources))
        self.assertFalse(RetrievalService._citation_ids_are_valid({"S4"}, sources))
        self.assertFalse(RetrievalService._citation_ids_are_valid(set(), sources))

    def test_insufficient_evidence_answer_detection(self) -> None:
        self.assertTrue(
            RetrievalService._is_insufficient_evidence_answer(
                "I do not have enough information to answer this question."
            )
        )
        self.assertTrue(
            RetrievalService._is_insufficient_evidence_answer(
                "There is insufficient evidence in the provided context."
            )
        )
        self.assertFalse(
            RetrievalService._is_insufficient_evidence_answer(
                "The model uses OCR for plate recognition [S1]."
            )
        )


if __name__ == "__main__":
    unittest.main()
