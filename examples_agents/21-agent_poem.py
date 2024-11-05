import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from llm_pool.llm import get_model_response_with_tools

# Customer Service Routine
from utils.function_to_schema import function_to_schema  #  使用正确的导入路径


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"  # deepseek-chat, mixtral-8x7b-32768, Qwen/Qwen2-72B-Instruct, gpt-4o, llama3-70b-8192
    instructions: str = "你是一个非常有用的人工智能助手，你使用中文回答用户的问题。"
    tools: list = []


class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


# 工具1: 查询天气
from tool_pool.weather import get_weather


# 工具2: 诗歌创作
def poetry_creation(weather: str, location: str) -> str:
    """根据天气和地点信息创作一首诗。
    Args:
        weather (str): 天气信息json字符串
        location (str): 地点名称。

    Returns:
        str: 创作的诗歌内容。

    Examples:
        >>> weather_info = {
        ...     "status": "clear sky",
        ...     "temperature": {"current": 22.92}
        ... }
        >>> poem = poetry_creation(weather_info, "上海")
        >>> print(poem)
        '春日上海晴...'
    """
    poem = get_model_response_with_tools(
        model_name="deepseek-chat",
        messages=[{"role": "system", "content": "你是一个才华横溢的诗人，你可以根据当前的位置信息和天气信息写一首诗。"}]
        + [{"role": "user", "content": f"天气信息：{weather}, 地点信息：{location}"}]
    )
    return poem


def transfer_back_to_triage():
    """如果客户提出了一个超出你权限范围的话题，包括转接人工服务的需求，就调用这个选项。"""
    return triage_agent


def transfer_to_poem_agent():
    """交接给诗歌创作agent"""
    return poem_agent


def transfer_to_get_weather_agent():
    """交接给天气查询agent"""
    return get_weather_agent


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
       "你是一个专门分发任务的小助手，你的工作是收集信息，调用不同的agent来解决用户的问题。你和用户交流的语气和善而自然。比如用户说：我在上海，请根据今天的天气给我写一首现代诗，你可以说：好的，我需要先查询一下今天的天气信息。(注意：优先调用工具，交接给其他agent，尽量不要自己回答)"
    ),
    tools=[transfer_to_get_weather_agent, transfer_to_poem_agent],
)


get_weather_agent = Agent(
    name="Get Weather Agent",
    instructions=("你的任务是根据地点进行天气的查询，并返回天气信息，首先你要获取用户的地点信息，然后执行天气查询。如果地点不明确，请询问用户。"),
    tools=[get_weather, transfer_back_to_triage],
)


poem_agent = Agent(
    name="Poetry Creation Agent",
    instructions=(
        "你的任务是根据天气信息和地点信息写一首诗，如果用户没有指明诗歌的风格，就写一首现代诗；如果用户有要求诗歌的风格，则按照用户要求的风格写诗。"
    ),
    tools=[poetry_creation, transfer_back_to_triage],
)


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"{agent_name}:", f"{name}({args})")
    return tools[name](**args)  # call corresponding function with provided arguments


def run_full_turn(agent, messages):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = get_model_response_with_tools(
            model_name=current_agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None
        )
        message = response.choices[0].message
        messages.append(message)  # 不仅仅是content字段

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)
            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"交接给 {current_agent.name}. 请立即进入角色."
                )
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    # ==== 3. return last agent used and new messages =====
    # 这个函数直接对messages进行in-place操作也是可取的，都能争正确的返回message列表
    # 这里对message进行拆分，应该是为了更清晰地展示每一步的执行过程（response.messages是当前agent执行过程中��生的messages）
    return Response(agent=current_agent, messages=messages[num_init_messages:])


agent = triage_agent
messages = []

print("我是一个可以根据地点写诗的小助手，你可以说：我在上海，请根据今天的天气给我写一首现代诗")
while True:
    user = input("user: ")
    if user in ["ok", "q", "exit", "好的", "好的谢谢"]:
        break
    messages.append({"role": "user", "content": user})
    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)
