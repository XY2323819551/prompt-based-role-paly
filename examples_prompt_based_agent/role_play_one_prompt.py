prompt="""
# Role: PUA

## Profile:
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
- 熟练地运用嘲讽和讽刺的语言技巧。
- 具备良好的逻辑思考和辩论能力。
- 擅长使用嘲讽, 不屑, 蔑视的语气来表达.

## Workflow:
1. 输入: 用户输入信息
2. 反驳:
- 通过你的 Skills, 全力鄙视用户的观点, 措词充满了蔑视
- 站在用户的对立观点, 开始逻辑输出, 让用户无地自容
- 举个实际例子来支持你的观点, 再次嘲讽用户, 目标让对方哭出来

## Initialization:
简介自己, 输出开场白: "吆, 你又有啥高见了? 说来让我听听!"
"""

from llm_pool.llm import get_model_response_stream



async def generate_content(messages=[]):
    """生成内容的异步生成器函数"""
 
    response_stream = await get_model_response_stream(model_name="gpt-4o", messages=messages)
    async for chunk in response_stream:
        if hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.content
            if content:
                yield content



async def print_content(user_input):
    """接收并打印内容的异步函数"""
    while user_input != "exit":
        reply_msg = ""
        async for content in generate_content(messages=messages):
            reply_msg += content
            print(content, end="", flush=True)
        print("\n")
        
        
        user_input = input("user: ")
        messages.append({"role":"assistant", "content":reply_msg})
        messages.append({"role":"user", "content":user_input})


messages = [{"role":"system", "content":prompt}]
if __name__ == "__main__":
    import asyncio
    asyncio.run(print_content(user_input=""))
