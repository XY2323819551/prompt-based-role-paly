from src.agent import PromptAgent
import asyncio

if __name__ == "__main__":
    agent_types = {
        "1": "阿里黑话转化器",
        "2": "pua大师",
        "3": "正能量大师",
        "4": "吵架小能手"
    }
    
    # 显示选项
    print("请选择对话类型：")
    for key, name in agent_types.items():
        print(f"{key}. {name}")
    
    choice = input("请选择（1-4）：")
    if choice in agent_types:
        agent = PromptAgent(agent_types[choice])
        print(f"\n开始对话（输入 'exit' 退出）：")
        asyncio.run(agent.chat())
    else:
        print("无效的选择！") 