import asyncio
import sys
from pathlib import Path
from typing import List, Dict
from duckduckgo_search import DDGS

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response

async def plan(question: str) -> str:
    plan_system_prompt = """
    Let's first understand the problem and devise a plan to solve the problem.
    Please output the plan starting with the header 'Plan:' and then followed by a numbered list of steps. 
    Please make the plan the minimum number of steps required to accurately complete the task. If the task is a question, 
    the final step should almost always be 'Given the above steps taken, please respond to the users original question'. 
    """

    plan_messages = [
        {'role': 'system', 'content': plan_system_prompt},
        {'role': 'user', 'content': question}
    ]

    plans = await get_model_response(model_name="deepseek-chat", messages=plan_messages)
    return plans

async def execute(query: str) -> List[Dict]:
    def internet_search(query: str):
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(
                query,
                max_results=5, 
                region="wt-wt", 
                safesearch="moderate", 
                timelimit="y",
                backend="api",
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return "No results found."

    return internet_search(query)

async def synthesize(question: str, search_results: List[Dict]) -> str:
    synthesis_system_prompt = "You are a helpful assistant that can analyze search results and answer questions."
    synthesis_user_prompt = f"""
    Based on the following search results and the original question, please provide a concise answer:

    Original question: {question}

    Search results:
    {search_results}

    Please synthesize the information and answer the original question.
    """

    synthesis_messages = [
        {'role': 'system', 'content': synthesis_system_prompt},
        {'role': 'user', 'content': synthesis_user_prompt}
    ]

    final_answer = await get_model_response(model_name="deepseek-chat", messages=synthesis_messages)
    return final_answer

async def plan_and_execute(question: str) -> str:
    # 1. Plan
    plans = await plan(question)
    print("Plan:")
    print(plans)

    # 2. Execute
    search_results = []
    search_results.append(await execute('the current population of Toronto'))
    search_results.append(await execute('the current population of New York city'))

    # 3. Synthesize results
    final_answer = await synthesize(question, search_results)
    return final_answer

async def main():
    question = 'What is the population gap between Toronto and New York city?'
    result = await plan_and_execute(question)
    print("Final answer:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
