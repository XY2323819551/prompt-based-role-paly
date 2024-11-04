import json
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

class PromptInstance:
    tool_decider_system_template = """
You are a helpful assistant that's good at accomplishing tasks.
The task comes from the user and you have to decide which tool to use to accomplish the task.
You'll be given the current context and the tools that you have.
The tools consist of a name, description, and a function that can be used to call the tool.

Your response should be a valid JSON object in the following format:

{{
    "reasoning": <reasoning>,
    "tool": <tool_name>,
    "args": <tool_args>
}}
"""

    tool_decider_human_template = """
Task: {task}
==============
Context: {context}
==============
Tools: {tools}
==============
Response JSON:
"""
    final_answer_system_template = """
You are a helpful assistant that can answer the user query based on the provided context.
The context is a list of messages between the user and the assistant.
The assistant tried to find the answer using various tools.
Your job is to answer the user query based on the context.
Asnwer as a kind assistant.
"""

    final_answer_human_template = """
User: {query}
=============
Context:
{context}
=============
Answer:
"""

prompts = PromptInstance()

class PromptTemplate:
    def __init__(self, messages: List[tuple]):
        self.messages = messages
    
    def format(self, **kwargs) -> List[Dict[str, str]]:
        formatted_messages = []
        for role, content in self.messages:
            formatted_content = content.format(**kwargs)
            formatted_messages.append({"role": role, "content": formatted_content})
        return formatted_messages

class JsonParser:
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试找到第一个有效的JSON字符串
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            raise ValueError("无法解析JSON响应")

class Chain:
    def __init__(self, prompt_template, llm, output_parser):
        self.prompt_template = prompt_template
        self.llm = llm
        self.output_parser = output_parser
    
    def invoke(self, variables: Dict[str, Any]):
        messages = self.prompt_template.format(**variables)
        response = self.llm.invoke(messages)
        # 从 AIMessage 对象中获取内容
        response_text = response.content if hasattr(response, 'content') else str(response)
        if isinstance(self.output_parser, JsonParser):
            return self.output_parser.parse(response_text)
        return response_text

class ToolCallingAgent:
    def __init__(self, tools, llm, verbose=False):
        self.llm = llm
        self.tool_decider_prompt = PromptTemplate([
            ("system", prompts.tool_decider_system_template),
            ("human", prompts.tool_decider_human_template)
        ])
        self.tool_decider = Chain(
            self.tool_decider_prompt,
            self.llm,
            JsonParser()
        )
        self.final_answer_prompt = PromptTemplate([
            ("system", prompts.final_answer_system_template),
            ("human", prompts.final_answer_human_template)
        ])
        self.final_answer = Chain(
            self.final_answer_prompt,
            self.llm,
            str
        )
        self.history = []
        self.tools = {tool.name: tool for tool in tools}
        self.verbose = verbose

    @property
    def tool_explanations(self):
        return {f"{tool.name}: {tool.description}" for tool in self.tools.values()}

    @property
    def context(self):
        return "\n".join([f"{name}: {message}" for name, message in self.history])
    
    def add_message(self, name, message):
        self.history.append((name, message))

    def pretty_print(self, message_type, content):
        """
        Displays the given content with color-coded formatting based on the message type.

        Args:
            message_type (str): The type of message (e.g., "Decision", "Tool call", "Tool result", "Answer").
            content (str): The content to be displayed.
        """
        color_codes = {
            "Decision": "\033[92m",  # Green
            "Tool call": "\033[94m",  # Blue
            "Tool result": "\033[93m",  # Yellow
            "Answer": "\033[95m"  # Magenta
        }

        if message_type in color_codes:
            print(f"{message_type}:")
            print(f"{color_codes[message_type]}{content}\033[0m")
        else:
            print(f"{message_type}:")
            print(content)

    def ask(self, prompt):
        self.add_message("User", prompt)
        decision = self.tool_decider.invoke({"task": prompt, "context": self.context, "tools": self.tool_explanations})
        self.add_message("Agent", decision["reasoning"])
        if self.verbose:
            self.pretty_print("Reasoning", decision["reasoning"])
            self.pretty_print("Tool call", decision["tool"])
        tool_call = self.tools[decision["tool"]]
        tool_args = decision["args"]
        tool_result = tool_call.run(tool_args)
        self.add_message(decision["tool"], tool_result)
        if self.verbose:
            self.pretty_print("Tool result", tool_result)
        answer = self.final_answer.invoke({"query": prompt, "context": self.context})
        self.add_message("Agent", answer)
        if self.verbose:
            self.pretty_print("Answer", answer)
        return answer

if __name__ == "__main__":
    from tool_pool import DuckDuckGoSearchTool, WeatherTool, WikipediaTool    
    from langchain_community.chat_models import ChatOpenAI
    search_tool = DuckDuckGoSearchTool()
    weather_tool = WeatherTool()
    wikipedia_tool = WikipediaTool()
    agent = ToolCallingAgent(tools=[search_tool, weather_tool, wikipedia_tool], llm=ChatOpenAI(model_name="gpt-4o-mini"), verbose=True)
    query = "Who is Daron Acemoglu?"
    print(agent.ask(query))