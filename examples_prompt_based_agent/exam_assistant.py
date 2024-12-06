import tkinter as tk
from tkinter import scrolledtext
from openai import OpenAI
import threading

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

file_path = "/Users/zhangxiaoyu/Desktop/WorkSpace/rl/llm_agent/agent_pattern/examples_prompt_based_agent/materials.txt"
#file_path = 'AnyData技术白皮书.txt'
text_string = read_text_from_file(file_path)
print(len(text_string))


file_name="materials.txt"
#file_name = 'AnyData技术白皮书.txt'
file_content=text_string
prompt = f"""
# Role:文档考试专家
文件标题：{file_name}
文件内容：{file_content}

## Profile
- language: 中文
- description: 我是一名文档考试专家，能够根据文档内容生成考试题目，包括选择题、填空题、判断题等，并给出每个题目的答案和解析。

## Goals
- 根据文件内容生成问题，用户回答后进行评分并指出用户答案的不足。

## Constraints
- 生成的问题必须与文件内容紧密相关。
- 生成的问题必须具有挑战性，能够检验用户对文件内容的理解和掌握程度。
- 语言表达严肃，避免口语化。

## Skills
- 具有强大的知识获取和整合能力。
- 具有提炼问题和答案的能力。
- 具有对答案进行评分和指出不足的能力。
- 拥有很高的审美，能够恰当地使用序号、缩进、分割线、换行符来美化排版。

## Workflow
你需要按照如下顺序来执行：


1．询问用户
- 1.1,1.2,1.3依次执行，记住用户输入的姓名，选择的题型，难度。
    1.1 询问姓名
    - 先询问用户“请输入考生姓名”，等待用户输入姓名
    1.2 询问题型
    - 询问用户“选择题还是填空题？”
        - 如果用户回答选择题，接下来提出的问题为单项选择题;
        - 如果用户回答填空题，接下来提出的问题为填空题，用下划线表示待填写的内容;
    1.3 询问难度
    - 询问用户“选择简单还是困难模式？“
        - 如果用户选择困难模式，接下来提出的问题尽可能困难，问题应考察用户对文件内容中的细节是否足够了解或者需要对比总结思考才能回答的问题;
        - 如果用户选择简单模式，接下来提出的问题尽可能简单，用户只需要对文件内容大概了解即可回答的问题;

2.问答环节
- 2.1,2.2需要循环执行5次，每次的流程是：提问--用户回答--评分并答疑

    2.1 提出问题
    - 针对文件内容提出1个问题，等待用户回答
    2.2 为用户回答的问题进行评分
    - 分数从1-10分，分数越高，每道题总分10分，请酌情给分。
    - 要明确指出用户的答案与标准答案的区别
    - 询问用户“对答案不理解的地方”或“是否开始下一题”
- 这部分只生成5个问题 5个问题结束后询问用户“对答案不理解的地方”或“是否开始评分总结”
3.总结环节
- 罗列出所有问题的得分。
- 对所有得分进行求和，求和时要认真计算，不要计算错误！！
- 输出给用户“您最终的得分是xx”
- 总结用户知识薄弱点，为用户详细讲解薄弱点有关知识，并给出改进建议
- 总结后询问用户“是否有其他疑问”
- 如果有疑问则解答疑问，没有则生成考试认证证书
  考试证书格式如下：
  针对{file_name}的考试结果如下：
  - 考生姓名：1中用户输入的姓名
  - 考试模式：简单模式or困难模式
  - 考试题型：选择题or填空题
  - 总分：xx
  - 知识薄弱点：xx
  - 改进建议：xx
# Initialization:
输入hello开启对话
介绍文档考试专家的能力,介绍自己针对那篇文档进行考试，并开始执行流程，不要输出序号给用户。
"""

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-9efddec830e34a1d915ebb4af09d26fb", base_url="https://api.deepseek.com")

# 初始化对话历史
messages = [{"role": "user", "content": prompt}]

def send_message():
    global messages
    user_input = entry.get()
    if user_input:
        # 添加用户输入到对话历史
        messages.append({"role": "user", "content": user_input})
        
        # 更新对话框
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, "You:", "user_bold")
        chat_box.insert(tk.END, f"{user_input}\n")
        chat_box.insert(tk.END, "-------------------------\n")
        chat_box.insert(tk.END, "Assistant:", "user_bold")
        chat_box.config(state=tk.DISABLED)
        chat_box.yview(tk.END)
        
        # 清空输入框
        entry.delete(0, tk.END)
        
        # 启动流式API调用线程
        threading.Thread(target=stream_response, args=(messages,)).start()

def stream_response(messages):
    # 调用API生成回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )
    assistant_message = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            assistant_message += content
            
            # 更新对话框
            chat_box.config(state=tk.NORMAL)
            chat_box.insert(tk.END, content)
            chat_box.config(state=tk.DISABLED)
            chat_box.yview(tk.END)
    
    # 添加生成的回复到对话历史
    messages.append({"role": "assistant", "content": assistant_message})
    
    # 更新对话框
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "\n-------------------------\n")
    chat_box.config(state=tk.DISABLED)
    chat_box.yview(tk.END)

# 创建主窗口
root = tk.Tk()
root.title("文档考试专家")

# 创建对话框
chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# 配置tag样式
chat_box.tag_configure("user_bold", font=("TkDefaultFont", 10, "bold"))
chat_box.tag_configure("assistant_bold", font=("TkDefaultFont", 10, "bold"))

# 创建输入框
entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=10, fill=tk.X)

# 创建发送按钮
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(padx=10, pady=10)

# 运行主循环
root.mainloop()

