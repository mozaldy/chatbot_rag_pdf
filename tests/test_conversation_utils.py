import unittest

from app.services.conversation_utils import (
    conversation_context_for_prompt,
    latest_user_query,
    normalize_conversation_messages,
)


class ConversationUtilsTests(unittest.TestCase):
    def test_normalize_from_single_string(self) -> None:
        normalized = normalize_conversation_messages("What is OCR?", max_messages=8)
        self.assertEqual(normalized, [{"role": "user", "content": "What is OCR?"}])

    def test_normalize_keeps_recent_messages_only(self) -> None:
        messages = [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
        ]
        normalized = normalize_conversation_messages(messages, max_messages=2)
        self.assertEqual(
            normalized,
            [
                {"role": "assistant", "content": "m2"},
                {"role": "user", "content": "m3"},
            ],
        )

    def test_latest_user_query_falls_back_to_last_content(self) -> None:
        messages = [
            {"role": "assistant", "content": "Here is context."},
        ]
        self.assertEqual(latest_user_query(messages), "Here is context.")

    def test_prompt_context_includes_user_and_assistant_only(self) -> None:
        messages = [
            {"role": "system", "content": "internal"},
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
        ]
        context = conversation_context_for_prompt(messages)
        self.assertNotIn("internal", context)
        self.assertIn("User: Question one", context)
        self.assertIn("Assistant: Answer one", context)


if __name__ == "__main__":
    unittest.main()
