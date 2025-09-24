import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append("../backend")
from main import call_llm_api

class TestNLQueryDashboard(unittest.TestCase):
    @patch("main.requests.post")
    def test_llm_query_priority(self, mock_post):
        # Simulate Open Source LLM fails, OpenAI returns a result
        mock_post.side_effect = [
            MagicMock(ok=False),
            MagicMock(ok=True, json=lambda: {"choices": [{"message": {"content": "42"}}]})
        ]
        result = call_llm_api("What is the answer?", ["a", "b"], [{"a": 1, "b": 2}])
        self.assertIn("choices", result)

if __name__ == "__main__":
    unittest.main()
