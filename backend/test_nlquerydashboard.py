import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from main import call_llm_api

class TestNLQueryDashboard(unittest.TestCase):
    @patch("main.requests.post")
    @patch("main.load_llm_config")
    def test_llm_query_priority(self, mock_load_llm_config, mock_post):
        # Patch LLM config to provide two providers: open_source_llm and openai
        mock_load_llm_config.return_value = [
            {"name": "open_source_llm", "url": "http://dummy1", "priority": 1},
            {"name": "openai", "url": "http://dummy2", "api_key": "sk-test", "priority": 2}
        ]
        # Simulate Open Source LLM fails, OpenAI returns a result
        mock_post.side_effect = [
            MagicMock(ok=False),
            MagicMock(ok=True, json=lambda: {"choices": [{"message": {"content": "42"}}]})
        ]
        result = call_llm_api("What is the answer?", ["a", "b"], [{"a": 1, "b": 2}])
        self.assertIn("choices", result)

if __name__ == "__main__":
    unittest.main()
