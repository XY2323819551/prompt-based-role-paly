import json
import os
from datetime import datetime
from typing import Dict, Any

class AgentLogger:
    def __init__(self, log_dir: str = "agent_logs"):
        self.log_dir = log_dir
        self.current_log = {
            "start_time": datetime.now().isoformat(),
        }
        self.current_round = 1
        self.init_new_round()
        self._ensure_log_dir()
        
    def _ensure_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def init_new_round(self):
        """初始化新的对话轮次"""
        round_key = f"round{self.current_round}"
        self.current_log[round_key] = {
            "conversation": [],
            "agents": []
        }
        self.current_agent = None
        
    def start_new_round(self):
        """开始新的对话轮次"""
        self.current_round += 1
        self.init_new_round()
            
    def log_user_message(self, content: str):
        """记录用户消息"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": content
        }
        round_key = f"round{self.current_round}"
        self.current_log[round_key]["conversation"].append(message)
            
    def log_agent_message(self, agent_name: str, content: str):
        """记录agent的回复"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": "agent",
            "agent_name": agent_name,
            "content": content
        }
        round_key = f"round{self.current_round}"
        self.current_log[round_key]["conversation"].append(message)
        
        # 同时记录到agent的steps中
        if self.current_agent:
            step = {
                "timestamp": datetime.now().isoformat(),
                "type": "llm_response",
                "content": content
            }
            self.current_agent["steps"].append(step)
            
    def start_agent_session(self, agent_name: str):
        """开始新的agent会话"""
        round_key = f"round{self.current_round}"
        
        # 如果当前agent已存在且有相同名称，直接使用它
        for agent in self.current_log[round_key]["agents"]:
            if agent["name"] == agent_name:
                self.current_agent = agent
                return
                
        # 否则创建新的agent记录
        self.current_agent = {
            "name": agent_name,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        self.current_log[round_key]["agents"].append(self.current_agent)
        
    def log_tool_call(self, tool_name: str, inputs: Dict[str, Any], output: Any):
        """记录工具调用"""
        if self.current_agent is None:
            raise ValueError("No active agent session")
            
        step = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_call",
            "tool_name": tool_name,
            "inputs": inputs,
            "output": str(output)
        }
        self.current_agent["steps"].append(step)
        
    def save_log(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_log_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.current_log, f, ensure_ascii=False, indent=2)
        
        return filepath 