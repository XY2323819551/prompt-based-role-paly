import re
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response
from utils.search import internet_search, process_content

class WebSearch:
    def __init__(self, name:str='web_search', threshold:int=8000):
        self.system_prompt = """
You are a Insight Researcher.

1. To find detail information for the user query
and summary the content into one sentence as simple as possible
2. If the user's question is about specific numerical values, 
only return the numerical results without any additional explanation.
"""
        self.name = name
        self.description = "the tool use for web search"
        self.threshold = threshold

    async def __call__(self, query:str):
        results = internet_search(query)
        all_text = ""
        windows_size = 0
        for item in results:
            if windows_size >= self.threshold:
                break
            page_content = process_content(item['href'])
            for page in page_content:
                if windows_size + len(page) > self.threshold:
                    remaining_space = self.threshold - windows_size
                    all_text += page[:remaining_space].strip() + "\n\n"
                    windows_size = self.threshold
                    break
                else:
                    windows_size += len(page)
                    all_text += page + "\n\n"
            if windows_size >= self.threshold:
                break

        msg = [{"role":"system","content":self.system_prompt},
               {"role":"user", "content": f"The query is {query}, The search results are {all_text}"}]
        
        answer = await get_model_response(model_name="deepseek-chat", messages=msg)
        return answer



async def format_message(messages: List[Dict]):
    formatted_json = json.dumps(messages, indent=4)
    print(formatted_json.replace('\\n', '\n'))
    print("-"*120)

async def react(question: str, tools: List[Callable]) -> str:
    react_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, return format like `action_input`
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
"""

    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    
    user_prompt = react_prompt.format(
        tools=tool_descriptions,
        tool_names=tool_names,
        input=question
    )

    messages = [{"role": "user", "content": user_prompt}]
    while True:
        await format_message(messages)
        response = await get_model_response(model_name="deepseek-chat", messages=messages, stop=["Observation", " Observation"])
        messages[-1]["content"] += response
        
        if "Final Answer:" in response:
            break
        
        regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, response, re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip(" ").strip('\n').strip('"')
            
            tool = next((t for t in tools if t.name == action), None)
            if tool:
                observation = await tool(action_input)
                messages[-1]["content"] += f"\nObservation: {observation}\nThought:"
    

    await format_message(messages)
    final_answer = re.search(r"Final Answer: (.*)", response, re.DOTALL)
    return final_answer.group(1).strip() if final_answer else "No final answer found."

async def main():
    query = "2024 UEFA European Championship and 2024 Copa América champion"
    search_tool = WebSearch()
    tools = [search_tool]
    
    result = await react(query, tools)
    print("Final answer:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())



"""
最后一轮的messages信息
[
    {
        "role": "user",
        "content": "
Answer the following questions as best you can. You have access to the following tools:

web_search: the tool use for web search

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [web_search]
Action Input: the input to the action, return format like `action_input`
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: 2024 UEFA European Championship and 2024 Copa Am\u00e9rica champion
Question: 2024 UEFA European Championship and 2024 Copa Am\u00e9rica champion
Thought: To answer this question, I need to find out the winners of the 2024 UEFA European Championship and the 2024 Copa Am\u00e9rica. These events are scheduled for 2024, so the results should be available by then. I will start by searching for the winner of the 2024 UEFA European Championship.

Action: web_search
Action Input: \"2024 UEFA European Championship winner\"


Observation: Spain.
Thought:The winner of the 2024 UEFA European Championship is Spain. Now, I need to find out the winner of the 2024 Copa Am\u00e9rica.

Action: web_search
Action Input: \"2024 Copa Am\u00e9rica winner\"


Observation: Argentina wins the 2024 Copa Am\u00e9rica title over Colombia with a late goal.
Thought:I now know the final answer.

Final Answer: The winner of the 2024 UEFA European Championship is Spain, and the winner of the 2024 Copa Am\u00e9rica is Argentina."
    }
]
"""