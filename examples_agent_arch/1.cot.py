import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response

MODELS = ["deepseek-chat", "mixtral-8x7b-32768", "Qwen/Qwen2-72B-Instruct", "gpt-4o"]

async def chain_of_thought(question: str, use_cot: bool = False, model_name: str = "deepseek-chat") -> str:
    """
    conclusion: 通过看结果，发现显示的使用step-by-step的提示，效果更好。
    """
    
    system_prompt = "You are a helpful assistant"

    if use_cot:
        question += "\nA: Let's think step by step."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    response = await get_model_response(model_name=model_name, messages=messages)
    return response

async def test_model(model_name: str) -> Dict[str, str]:
    results = {}

    questions = [
        ("Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?", False),
        ("Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?", True),
        ("The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?", False),
        ("""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?""", False),
        ("""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step. Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?""", True),
        ("Q: 9.11 and 9.8, which is bigger?", False),
        ("Q: 9.11 and 9.8, which is bigger?", True),
        ("""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: 9.11 and 9.8, which is bigger?""", False),
        ("""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step. Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: 9.11 and 9.8, which is bigger?""", True)
    ]

    for i, (question, use_cot) in enumerate(questions, 1):
        result = await chain_of_thought(question, use_cot, model_name)
        results[f"result{i}"] = result

    return results

async def main():
    all_results = {}

    for model in MODELS:
        print(f"Testing model: {model}")
        try:
            model_results = await test_model(model)
            all_results[model] = model_results
            print(f"Completed testing for {model}")
        except Exception as e:
            print(f"Error testing {model}: {str(e)}")
        print("-" * 50)

    # 将结果保存到JSON文件
    output_file = Path(__file__).parent / "cot_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
