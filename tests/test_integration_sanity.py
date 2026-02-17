import unittest
from app.services.retrieval_service import RetrievalService

class TestIntegrationSanity(unittest.IsolatedAsyncioTestCase):
    async def test_chit_chat_flow(self):
        """
        Verifies that a Chit-Chat query returns a direct response without hitting the vector DB.
        """
        service = RetrievalService()
        # This should trigger the CHIT_CHAT route in the router
        response = await service.answer_query("Hello, who are you?")
        
        print(f"\n[ChitChat Response]: {response.get('response')}")
        
        # Assertions
        self.assertIn("response", response)
        self.assertTrue(len(response["response"]) > 0)
        # Chit-Chat should have NO sources
        self.assertEqual(response.get("sources"), [])

if __name__ == "__main__":
    unittest.main()
