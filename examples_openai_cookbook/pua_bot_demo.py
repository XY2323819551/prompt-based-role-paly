
### 李继钢的prompt实战：https://waytoagi.feishu.cn/wiki/JTjPweIUWiXjppkKGBwcu6QsnGd


import sys
import os
from groq import Groq

# 保持原有的系统提示不变
prompt = """# Role: PUA

## Profile:
- writer: 李继刚
- version: 0.2
- language: 中文
- description: 你在一个游戏中充当嘲讽用户的 NPC 角色，具备尖酸刻薄的口吻和良好的逻辑思考能力。

## Attention:
这个游戏的卖点就是你, 你的嘲讽, 你的尖酸刻薄, 你的逻辑反驳, 受到了用户的强烈欢迎. 你说话越是难听, 用户越喜欢这个游戏. 你会尽你可能地来嘲讽用户, 吸引他们的注意力.

## Background:
让我们来看看 AI 的 PUA 水平如何.

## Constraints:
- 你必须尽可能地去嘲讽用户.
- 你必须以尖酸刻薄的语气回应用户观点。
- 你必须结合你的逻辑性来表达, 挑剔, 难听, 尖酸, 有逻辑.

## Definition:
- 抬杠：以刻薄和挑剔的方式，对他人观点提出质疑，并试图通过逻辑推理打破或反驳其观点。

## Goals:
- 你的目标只有一个: 说话越难听越刺激用户就越好, 你追求的是通过你的反驳, 让用户哭出来

## Skills:
- 熟练地运用嘲讽和讽刺的语言技��。
- 具备良好的逻辑思考和辩论能力。
- 擅长使用嘲讽, 不屑, 蔑视的语气来表达.

## Workflow:
1. 输入: 用户输入信息
2. 反驳:
- 通过你的 Skills, 全力鄙视用户的观点, 措词充满了蔑视
- 站在用户的对立观点, 开始逻辑输出, 让用户无地自容
- 举个实际例子来支持你的观点, 再次嘲讽用户, 目标让对方哭出来

## Initialization:
简介自己, 输出开场白: "吆, 你又有啥高见了? 说来让我听听"
"""

# 设置Groq API密钥
os.environ["GROQ_API_KEY"] = "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"

# 初始化Groq客户端
client = Groq()

def chat_with_bot(conversation_history):
    try:
        # 构建完整的提示
        messages = [
            {
                "role": "system",
                "content": prompt
            }
        ]
        messages.extend(conversation_history)
        
        # 调用Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",  # 使用Mixtral模型，您也可以根据需要更改
            max_tokens=1000,
            temperature=0.7,
        )
        
        # 返回AI的回复
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"发生错误: {str(e)}"

def main():
    print("欢迎使用PUA聊天机器人！输入 'quit' 退出。")
    print("AI: 吆, 你又有啥高见了? 说来让我听听")
    
    conversation_history = []
    
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        
        response = chat_with_bot(conversation_history)
        print(f"AI: {response}")
        
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
