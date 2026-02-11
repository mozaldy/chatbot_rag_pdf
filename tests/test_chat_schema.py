import unittest

from app.models.schemas import ChatRequest


class ChatSchemaTests(unittest.TestCase):
    def test_chat_request_accepts_single_string(self) -> None:
        payload = ChatRequest(messages="What is the model?")
        self.assertEqual(payload.messages, "What is the model?")

    def test_chat_request_accepts_message_list(self) -> None:
        payload = ChatRequest(
            messages=[
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow up"},
            ]
        )
        assert isinstance(payload.messages, list)
        self.assertEqual(payload.messages[-1].content, "Follow up")


if __name__ == "__main__":
    unittest.main()
