OpenAI的Swarm 在5和6的基础上，又增加了哪些东西？

1. **context_variables**（没太理解这个东西的作用）和 Instructions 配合使用
2. **max_turns**（控制对话轮数，可以根据自己的需要灵活设置，比如超过最大轮数之后要做什么额外的处理）
3. **model_override**（切换当前agent的模型，也有很多方式可以实现）
4. **debug**（控制日志的一种方式而已，可以有很多种方式实现）
5. **Agent中定义的functions、tool_choice的区别**
    - functions 是工具的定义，tool_choice 是工具的选择
    - ```python
      # 自动选择 - 模型自行决定是否使用工具
      tool_choice = "auto"  

      # 强制使用指定的工具
      tool_choice = {
          "type": "function",
          "function": {"name": "get_weather"}  # 强制使用get_weather函数
      }

      # 不使用任何工具
      tool_choice = "none"
      ```
    - swarm的返回格式（用户根据自己的需求进行定义就可以）
    - ```python
      # 自动选择 - 模型自行决定是否使用工具
      messages=[
        {
            'content': None, 
            'refusal': None, 
            'role': 'assistant', 
            'function_call': None, 
            'tool_calls': [
                {
                    'id': 'call_LeOo5aU6F2UJkqoSzUz3rvtX', 
                    'function': {
                        'arguments': '{}', 
                        'name': 'transfer_to_chinese_agent'
                    }, 'type': 'function'
                }
            ], 
            'sender': 'English Agent'
        }, 
        {
            'role': 'tool', 
            'tool_call_id': 'call_LeOo5aU6F2UJkqoSzUz3rvtX', 
            'tool_name': 'transfer_to_chinese_agent', 
            'content': '{"assistant": "chinese Agent"}'
        }, 
        {
            'content': 'SOP 是“标准操作程序”（Standard Operating Procedure）的缩写。它是描述某一特定任务或过程的操作步骤和要求的文件。这种程序通常被用来确保在执行任务时的一致性和效率，减少错误，提高质量，并且可以用于培训员工。SOP 详细阐述了每个步骤中应采取的具体行动，以及在某些情况下应如何应对突发状况或异常状况。它们在各种行业中都被广泛使用，包括制造业、医药、食品服务、IT 和政府部门等。', 
            'refusal': None, 
            'role': 'assistant', 
            'function_call': None, 
            'tool_calls': None, 
            'sender': 'chinese Agent'
        }
      ]

        agent=Agent(
            name='chinese Agent', 
            model='gpt-4o', 
            instructions='You only speak chinese.', 
            functions=[],
            tool_choice=None, 
            parallel_tool_calls=True
        ) 

        context_variables={}

      ```
