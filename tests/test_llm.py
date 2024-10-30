import unittest
import asyncio
from llm_pool.llm import get_model_response

MODELS = ["deepseek-chat", "Qwen/Qwen2-72B-Instruct"]

class TestLLMResponses(unittest.TestCase):
    """Test cases for LLM response functionality"""

    def setUp(self):
        """Set up test cases"""
        self.messages = [
            {
                "role": "user",
                "content": "一加一等于几，直接给出答案，不要解释",
            }
        ]

    async def async_test_model(self, model_name):
        """Test individual model response"""
        try:
            response = await get_model_response(model_name, self.messages)
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            print(f"\nResponse from {model_name}:")
            print(response)
            return True
        except Exception as e:
            print(f"\nError testing {model_name}: {str(e)}")
            return False

    def test_all_models(self):
        """Test responses from all models"""
        async def run_all_tests():
            results = []
            for model in MODELS:
                print(f"\nTesting model: {model}")
                print("-" * 50)
                result = await self.async_test_model(model)
                results.append((model, result))
            return results

        # Run async tests
        results = asyncio.run(run_all_tests())
        
        # Check results
        for model, success in results:
            with self.subTest(model=model):
                self.assertTrue(success, f"Model {model} test failed")

if __name__ == '__main__':
    unittest.main(verbosity=2)
