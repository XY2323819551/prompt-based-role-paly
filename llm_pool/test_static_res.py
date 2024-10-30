import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response



MODELS = ["deepseek-chat", "mixtral-8x7b-32768", "Qwen/Qwen2-72B-Instruct", "gpt-4o"]

# test
import asyncio
async def test_get_model_response():
    messages=[
            {
                "role": "user",
                "content": "一加一等于几，直接给出答案，不要解释",
            }
        ]
    for model in MODELS:
        try:
            print(f"Testing model: {model}")
            response = await get_model_response(model, messages)
            print(f"Response from {model}:")
            print(response)
            print("-" * 50)
        except Exception as e:
            print(f"Error testing {model}: {str(e)}")
            print("-" * 50)

def run_test():
    asyncio.run(test_get_model_response())

if __name__ == "__main__":
    run_test()
