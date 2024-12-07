# Prompt-based Toy Agent

一个基于大语言模型的多角色对话系统，支持多种预设角色进行对话。prompt的内容来自李继刚。

## 项目介绍

本项目是一个灵活的多角色对话系统，通过预设不同的角色提示词（Prompt），让用户可以与不同角色的AI进行对话。目前支持以下角色：

- 阿里黑话转化器：将普通话转换为阿里巴巴特色的企业黑话
- PUA大师：一个尖酸刻薄的对话角色
- 正能量大师：将消极的词汇转换为积极的表达
- 吵架小能手：擅长抬杠和辩论的对话角色

## 项目结构

```
.
├── src/
│   ├── config.py      # 配置文件，包含模型配置和提示词
│   ├── agent.py       # 对话代理的核心实现
│   └── llm.py         # 大语言模型接口封装
├── main.py            # 主程序入口
├── .env              # 环境配置文件
└── requirements.txt   # 项目依赖
```

## 运行方法

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   在项目根目录创建 `.env` 文件，配置以下内容：
   ```bash
   # OpenAI API配置
   OPENAI_API_KEY=你的OpenAI API密钥
   # 其他模型配置（如果使用其他模型）
   DEEPSEEK_API_KEY=你的Deepseek API密钥
   ```

3. 运行程序：
   ```bash
   python main.py
   ```

4. 使用说明：
   - 程序启动后会显示可用的角色列表
   - 输入对应的数字（1-4）选择角色
   - 开始对话：直接输入文字与AI对话
   - 结束对话：输入 'exit'、'quit'、'q' 或 'bye'

## 扩展方法

要添加新的对话角色，只需要两步：

1. 在 `src/config.py` 中添加新的 prompt 和映射：
   ```python
   # 添加新的 prompt
   NEW_ROLE_PROMPT = """
   # Role: 新角色名称

   ## Profile:
   - language: 中文
   - description: 角色描述

   ## Goals:
   - 目标1
   - 目标2

   ## Skills:
   - 技能1
   - 技能2

   ## Workflows:
   1. 工作流程1
   2. 工作流程2
   """

   # 在 PROMPT_MAPPING 中添加映射
   PROMPT_MAPPING = {
       "阿里黑话转化器": CANT_CONVERTER_PROMPT,
       "pua大师": PUA_PROMPT,
       "正能量大师": POSITIVE_ENERGY_PROMPT,
       "吵架小能手": LITTLE_BRAWLER_PROMPT,
       "新角色": NEW_ROLE_PROMPT,  # 添加新角色
   }
   ```

2. 在 `main.py` 中更新选项菜单：
   ```python
   agent_types = {
       "1": "阿里黑话转化器",
       "2": "pua大师",
       "3": "正能量大师",
       "4": "吵架小能手",
       "5": "新角色",  # 添加新选项
   }
   ```

完成这两步后，新的对话角色就添加完成了，无需修改其他代码。

## 示例对话

```
请选择对话类型：
1. 阿里黑话转化器
2. pua大师
3. 正能量大师
4. 吵架小能手

请选择（1-4）：1

开始对话（输入 'exit' 退出）：
user: 找个小众产品抄
assistant: 找准了自己差异化赛道，通过精细化运营实现价值倍增

user: exit
```

## 注意事项

- 需要确保已安装 Python 3.7 或更高版本
- 必须配置正确的API密钥才能使用
