import re
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Sequence, Union, Callable, Collection, Optional

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
               {"role":"user", "content": f"The search query {query}\nThe search results are {all_text}"}]
        
        answer = await get_model_response(model_name="deepseek-chat", messages=msg)
        return answer

class AbsDifferenceCode:
    """
    This tool is used to calculate the absolute difference between two numerical values.

    """
    def __init__(self, name:str="math_code"):
        self.system_prompt = """Extract the numerical values from the strings of the minuend and subtrahend provided by the user."""

        self.name = name
        self.description = "math tool for calculating subtractions using python code. more accurate than llm."
        
    async def __call__(self, concat_content:str) -> float:
        bracket_content = concat_content.split('[')[1].split(']')[0]  # 先分割得到中括号部分
        items = bracket_content.split("',")  # 再分割得到两个内容
        result1 = items[0].strip().strip("'")  # 清理字符串并赋值
        result2 = items[1].strip().strip("'")
        user_content = f"""
input parameters
<minuend>
{result1}
</minuend>

<subtrahend>
{result2}
</subtrahend>

output parameters in json format
```josn
{{
    "minuend":"the value of minuend in float",
    "subtrahend": "the value of subtrahend in float"
}}
```

"""
        messages = [
            {"role":"system", "content":self.system_prompt},
            {"role":"user", "content":user_content}
        ]
        response = await get_model_response(model_name="deepseek-chat", messages=messages, is_json=True)
        result_dict = json.loads(response)
        
        # calculate the absolute difference
        minuend = result_dict["minuend"]
        subtrahend = result_dict["subtrahend"]
        if minuend >= subtrahend:
            return minuend - subtrahend
        else:
            return subtrahend - minuend

class AbsDifferenceLLM:
    def __init__(self, name:str="math_llm"):
        self.system_prompt = "You are a calculation assistant."
        self.name = name
        self.description = "math tool for calculating subtractions using llm."
        
    async def __call__(self, question:str) -> str:
        user_content = f"""
Answer the Question based on the Context. When you write down a expression, it MUST ONLY consists of numbers and operators. Here are some guidelines that you will be PANALIZED if you don't follow:

  - When you are asked for differences, you consider the absolute value of the difference. Difference of two numbers is always positive.For instance, the difference between 1 and 2 is 1, not -1.
  - When you are applying operations (e.g. difference, summation, ratio, etc.) between multiple values in the Context, you must unify the units of those numbers. For instance, you cannot add 1 meter to 1 foot.
     - You must pick the values in the same units if all the values are available in the same units.
     - If not, you must convert them to the same units before applying the operation.
  - You MUST strictly follow the unit (e.g. meter, kilometer, million, etc.) you were asked.
     - If the Context has the numbers in same units as the question, you can directly use them.
     - If the Context has the numbers in different units than the question, you must convert them to the units asked in the question.For example, if the question asks for the distance between two cities in kilometers, but the Context has the distance in miles, you must convert the distance to kilometers.
  - If you are asked about a particular number in millions, billions, or any other unit, the number should be written without specifying the unit. For example, if you are asked for 100 millions, it should be written as 100, not 100 million or 100,000,000.
 - Never introduce a variable. For instance "gazelle_max_speed * 1.4" is not allowed. Pick up a correct number from the given context.

 Question: {question}
 
"""
        messages = [
            {"role":"system", "content":self.system_prompt},
            {"role":"user", "content":user_content}
        ]

        response = await get_model_response(model_name="deepseek-chat", messages=messages)
        return response

class Step:
    def __init__(self, 
                 idx: int, 
                 name: str, 
                 tool: callable,
                 args: Collection[Any],
                 dependencies: Collection[int]):
        self.idx = idx
        self.name = name
        self.tool = tool
        self.args = args
        self.dependencies = dependencies
        self.observation = None

    async def exec(self):
        self.observation = await self.tool(self.args)
        return self.observation

async def plan(question: str, tools: List[Callable]) -> str:
    system_prompt = f"""
Let's first understand the problem and devise a plan to solve the problem.
Please output the plan starting with the header 'Plan:' and then followed by a numbered list of steps. 
Please make the plan the minimum number of steps required to accurately complete the task. If the task is a question, 
the final step should almost always be 'Given the above steps taken, please respond to the users original question'. 

Each plan should comprise an action from the following {len(tools) + 1} types:
"""

    for i, tool in enumerate(tools):
        system_prompt += f"{i+1}. {tool.name}: {tool.description}\n"

    # few shot
    EXAMPLE_PROMPT = (
        "Question: If cheetah was 1.3 times slower, greyhound was 1.5 times faster, and falcon was 2.3 time slower then their maximum speeds, "
        "what will be the ratio of the fastest animal to the slowest animal?\n"
        '1. search("cheetah")\n'
        '2. math_llm("cheetah max speed in km/h if 1.3 times slower?", ["$1"]\n'
        '3. search("greyhound")\n'
        '4. math_llm("greyhound max speed in km/h if 1.5 times faster?", ["$3"]\n'
    '5. search("falcon")\n'
    '6. math_llm("falcon max speed in km/h if 2.3 times slower?", ["$5"]\n'
    '7. math_llm("max($2, $4, $6) / min($2, $4, $6)")\n'
    "Thought: I can answer the question now.\n"
    "###\n"
    "\n"
    "Question: If Mount Everest's height were halved and Mount Kilimanjaro's height were doubled, what would be the difference in their height?\n"
    "1. search('Mount Everest')\n"
    '2. math_llm("half of Mount Everest height in meter?", ["$1"])\n'
    '3. search("Mount Kilimanjaro")\n'
    '4. math_llm("double of Mount Kilimanjaro height in meter?", ["$3"])\n'
    '5. math_llm("abs($3 - $4)")\n'
    "Thought: I can answer the question now.\n"
    "###\n"
    "\n"
    "Question: With the Sahara Desert's area reduced by 33% and the Kalahari Desert's area magnified by 52%, which one covers more ground?\n"
    "1. search('Sahara Desert')\n"
    "2. math_llm('Sahara Desert area in km^2 if reduced by 33%?', ['$1'])\n"
    '3. search("Kalahari Desert")\n'
    "4. math_llm('Kalahari Desert area in km^2 if magnified by 52%?', ['$3'])\n"
    "Thought: I can compare the numbers without calling math.\n"
    "###\n"
    "\n"
    "Question: Determine the smaller value: the depth difference in meters between the Mariana Trench and the Puerto Rico Trench, "
    "or the depth difference in meters between the South Sandwich Trench and the Sunda Trench.\n"
    "1. search('Mariana Trench')\n"
    "2. search('Puerto Rico Trench')\n"
    "3. math_llm('absolute depth difference between Mariana and Puerto Rico Trench in meters?', ['$1', '$2'])\n"
    "4. search('South Sandwich Trench')\n"
    "5. search('Sunda Trench')\n"
    "6. math_llm('absolute depth difference between South Sandwich and Sunda Trench in meters?', ['$4', '$5'])\n"
    "7. math_llm('min($3, $6)')\n"
    "Thought: I can answer the question now.\n"
    "###\n"
    "\n"
    "Question: What is the raio of the height of Mount Everest and the height of Mount Kilimanjaro?\n"
    "1. search('Mount Everest')\n"
    "2. search('Mount Kilimanjaro')\n"
    "3. math_llm('height of Mount Everest / height of Mount Kilimanjaro', ['$1', '$2'])\n"
    "Thought: I can answer the question now.\n"
    "###\n"
)

    # the ahove prompt is refer to llm compiler
    system_prompt = system_prompt + '\n' + EXAMPLE_PROMPT

    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]

    response = await get_model_response(model_name="deepseek-chat", messages=messages)
    return response

async def plan_preset(question: str) -> str:
    """
    预先设置的plan，用于测试
    """
    plans_math_code= """
Plan:
1. search('Toronto population 2023')
2. search('New York City population 2023')
3. math_code('population difference between New York City and Toronto', ['$1', '$2'])
Thought: I can answer the question now.
"""
    plans_math_llm = """
Plan:
1. search('Toronto population 2023')
2. search('New York City population 2023')
3. math_llm('population difference between New York City and Toronto', ['$1', '$2'])
Thought: I can answer the question now.
"""
    return plans_math_code

async def execute(plans: str, tools: List[Callable]) -> List[str]:
    ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
    # 这个正则表达式包含四个捕获组：
    # 1. (\d+) 捕获步骤编号
    # 2. (\w+) 捕获函数名
    # 3. (.*) 捕获函数参数
    # 4. (\s*#\w+\n)? 是可选的，用于捕获可能存在的注释
    #
    # 最后一个可选的捕获组用于以下情况：
    # 1. 带注释的计划步骤，例如：
    #    1. search('Toronto population 2023') #城市人口查询
    # 2. 调试信息，例如：
    #    2. search('New York City population 2023') #需要验证
    # 3. 元数据或标签，例如：
    #    3. math_llm('population difference', ['$1', '$2']) #精确计算
    # 4. 兼容性考虑：确保正则表达式能够匹配包含或不包含注释的不同格式的计划
    #
    # 注意：如果确定不需要处理注释，可以简化正则表达式为：
    # ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)"
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def default_dependency_rule(idx, args: str):
        """
        检查给定的索引是否存在于参数字符串中的数字引用中。

        此函数用于确定某个动作或步骤（由idx表示）是否被其他步骤引用或依赖。
        它可能是规划和执行系统的一部分，用于确定任务之间的依赖关系。

        Args:
            idx (int): The integer index to check.
            args (str): The string containing potential numeric references.

        Returns:
            bool: True if idx is found in the numeric references within args, False otherwise.

        Note:
            - 函数使用ID_PATTERN正则表达式来匹配形如${数字}或$数字的模式。
            - 函数中包含breakpoint()调用，用于调试目的。在实际运行时，这会暂停程序执行。
        """
        matches = re.findall(ID_PATTERN, args)
        numbers = [int(match) for match in matches]
        return idx in numbers

    def find_tool(tool_name:str, tools: Sequence[Callable]=tools):
        for tool in tools:
            if tool.name == tool_name:
                return tool

    matches = re.findall(ACTION_PATTERN, plans)
    steps = []

    for item in matches:
        idx, tool_name, tool_args, _ = item
        idx = int(idx)
        tool = find_tool(tool_name)
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, tool_args)]
        step = Step(
            idx=idx,
            name=tool_name,
            tool=tool,
            args=tool_args,
            dependencies=dependencies
        )
        steps.append(step)

    results = []
    for step in steps:
        for dependency in sorted(step.dependencies, reverse=True):
            for arg_mask in ["${" + str(dependency) + "}", "$" + str(dependency)]:
                if arg_mask in step.args:
                    if steps[dependency-1].observation is not None:
                        step.args = step.args.replace(
                            arg_mask, str(steps[dependency-1].observation)
                        )
        result = await step.exec()
        results.append(result)
        print(f"Step {step.idx} executed: {result}")
    return results

async def synthesize(question: str, execution_results: List[str]) -> str:
    synthesis_system_prompt = "You are a helpful assistant that can analyze execution results and answer questions."
    synthesis_user_prompt = f"""
    Based on the following execution results and the original question, please provide a concise answer:

    Original question: {question}

    Execution results:
    {execution_results}

    Please synthesize the information and answer the original question.
    """

    breakpoint()
    synthesis_messages = [
        {'role': 'system', 'content': synthesis_system_prompt},
        {'role': 'user', 'content': synthesis_user_prompt}
    ]

    final_answer = await get_model_response(model_name="deepseek-chat", messages=synthesis_messages)
    return final_answer

async def plan_and_execute(question: str, need_synthesize: bool=True) -> str:
    # define tools
    search_tool = WebSearch(name="search")
    math_llm_tool = AbsDifferenceLLM(name="math_llm")
    math_code_tool = AbsDifferenceCode(name="math_code")
    tools = [search_tool, math_llm_tool, math_code_tool]

    # 1. Plan
    plans = await plan(question, tools)
    print(f"Plan:\n{plans}")

    # 2. Execute
    execution_results = await execute(plans, tools)

    # 3. Synthesize results
    if need_synthesize:
        final_answer = await synthesize(question, execution_results)
    else:
        final_answer = execution_results[-1]
    return final_answer

async def main():
    question = 'What is the population gap between Toronto and New York city?'
    result = await plan_and_execute(question, need_synthesize=False)
    print("Final answer:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
