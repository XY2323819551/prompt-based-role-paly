import os
from dotenv import load_dotenv
import json
import time
from typing import List, Dict, Any, Optional, Deque
from dataclasses import dataclass
from collections import deque
from zhipuai import ZhipuAI

load_dotenv()

class ConversationMemory:
    """自定义对话记忆管理器"""
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.messages: Deque[Dict[str, Any]] = deque(maxlen=max_size)
    
    def add_user_message(self, text: str):
        """添加用户消息"""
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}]
        })
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.messages.append({
            "role": "assistant",
            "content": content
        })
    
    def add_tool_message(self, content: str):
        """添加工具调用结果"""
        self.messages.append({
            "role": "tool",
            "content": content
        })
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """获取所有记忆中的消息"""
        return list(self.messages)
    
    def clear(self):
        """清空记忆"""
        self.messages.clear()

class GLM4AllToolsAgent:
    def __init__(
        self,
        api_key: str,
        tools: List[Dict] = None,
        memory_size: int = 10,
        temperature: float = 0.7,
        role_desc: str = None
    ):
        self.client = ZhipuAI(api_key=api_key)
        self.tools = tools or [{"type": "code_interpreter"}]
        self.memory = ConversationMemory(max_size=memory_size)
        self.temperature = temperature
        
        # 设置系统消息
        if role_desc:
            self.memory.add_user_message(role_desc)

    def _extract_tool_calls(self, response_content: str) -> Optional[Dict]:
        """从响应中提取工具调用信息"""
        if 'arguments=' in response_content and 'name=' in response_content:
            try:
                # 提取工具名称和参数
                parts = response_content.split(', name=')
                arguments = parts[0].replace('arguments=', '').strip("'")
                name = parts[1].strip("'")
                return {
                    "name": name,
                    "arguments": json.loads(arguments)
                }
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                return None
        return None

    def _collect_stream_response(self, response) -> str:
        """收集流式响应的内容"""
        full_response = ""
        for chunk in response:
            breakpoint()
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    print(content, end="", flush=True)
        print()  # 添加换行
        return full_response

    def chat(self, message: str) -> str:
        """与用户进行对话"""
        # 添加用户消息到记忆
        self.memory.add_user_message(message)
        
        # 调用 GLM-4-AllTools（使用流式调用）
        response = self.client.chat.completions.create(
            model="glm-4-alltools",
            messages=self.memory.get_messages(),
            tools=self.tools,
            temperature=self.temperature,
            stream=True  # 启用流式调用
        )
        
        # 收集流式响应内容
        assistant_message = self._collect_stream_response(response)
            
        # 检查是否包含工具调用
        tool_call = self._extract_tool_calls(assistant_message)
        if tool_call:
            # 添加助手的工具调用消息
            self.memory.add_assistant_message(assistant_message)
            
            # 这里应该实际调用工具并获取结果
            tool_result = "[100,100,200,200,300,400]"  # 实际应该是调用工具的返回值
            
            # 添加工具调用结果
            self.memory.add_tool_message(tool_result)
            
            # 再次调用 GLM-4 处理工具调用结果
            response = self.client.chat.completions.create(
                model="glm-4-alltools",
                messages=self.memory.get_messages(),
                tools=self.tools,
                temperature=self.temperature,
                stream=True  # 启用流式调用
            )
            
            assistant_message = self._collect_stream_response(response)
        
        # 添加最终的助手回复
        self.memory.add_assistant_message(assistant_message)
        return assistant_message

    def reset(self):
        """重置对话历史"""
        self.memory.clear()

def main():
    """测试主函数"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("在 .env 文件中未找到 ZHIPUAI_API_KEY")

    # 定义工具列表
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_tourist_data_by_year",
                "description": "用于查询每一年的全国出行数据，输入年份范围(from_year,to_year)，返回对应的出行数据。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "description": "交通方式，默认为by_all，火车=by_train，飞机=by_plane，自驾=by_car",
                            "type": "string"
                        },
                        "from_year": {
                            "description": "开始年份，格式为yyyy",
                            "type": "string"
                        },
                        "to_year": {
                            "description": "结束年份，格式为yyyy",
                            "type": "string"
                        }
                    },
                    "required": ["from_year","to_year"]
                }
            }
        },
        {
            "type": "code_interpreter"
        }
    ]

    agent = GLM4AllToolsAgent(
        api_key=api_key,
        tools=tools,
        role_desc="你是一个专业的数据分析助手"
    )

    try:
        # 测试查询
        query = "帮我查询2018年至2024年，每年五一假期全国旅游出行数据，并绘制成柱状图展示数据趋势。"
        print(f"\nUser: {query}")
        response = agent.chat(query)
        print(f"\nAssistant: {response}\n")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        agent.reset()

if __name__ == "__main__":
    main()
