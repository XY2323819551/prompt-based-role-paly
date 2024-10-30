"""
函数到 JSON Schema 转换工具。

这个模块提供了将 Python 函数转换为 JSON Schema 的功能，主要用于：
1. 解析函数的签名和文档字符串
2. 生成符合 OpenAI Function Calling 格式的 JSON Schema
3. 支持类型注解和参数描述的提取

主要功能：
- function_to_schema: 将 Python 函数转换为 JSON Schema
- parse_docstring: 解析函数的 docstring，提取描述和参数信息

使用示例：
    >>> def greet(name: str, age: int = 0):
    ...     '''Say hello to someone.
    ...     
    ...     Args:
    ...         name (str): The person's name
    ...         age (int, optional): The person's age
    ...     '''
    ...     pass
    >>> schema = function_to_schema(greet)
    >>> print(schema['function']['name'])
    'greet'

注意事项：
- 函数必须有有效的类型注解
- docstring 必须遵循 Google 风格
- 支持可选参数和默认值
"""

import re
import json
import inspect
import requests 
from bs4 import BeautifulSoup

def parse_docstring(docstring):
    """
    解析函数的 docstring，提取函数描述和参数描述。

    Args:
        docstring (str): 函数的 docstring

    Returns:
        tuple: (函数描述, 参数描述的字典)
    """
    if not docstring:
        return "", {}

    # 分割 docstring
    parts = docstring.split("Args:")
    
    # 获取函数主描述
    main_description = parts[0].strip()
    
    # 解析参数描述
    param_descriptions = {}
    if len(parts) > 1:
        # 提取 Args 部分到下一个主要部分（Returns:, Raises:, Examples:, Note: 等）
        args_section = parts[1].split('\n\n')[0]
        # 解析每个参数
        param_patterns = re.finditer(r'(\w+)\s*\(([\w\s,]+)\):\s*([^\n]+)', args_section)
        for match in param_patterns:
            param_name = match.group(1)
            param_description = match.group(3).strip()
            param_descriptions[param_name] = param_description

    return main_description, param_descriptions

def function_to_schema(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    # 解析 docstring
    main_description, param_descriptions = parse_docstring(func.__doc__)

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        
        param_info = {
            "type": param_type
        }
        
        # 添加参数描述（如果存在）
        if param.name in param_descriptions:
            param_info["description"] = param_descriptions[param.name]

        parameters[param.name] = param_info

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": main_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def test_basic_function():
    def basic_function(arg1, arg2):
        return arg1 + arg2

    result = function_to_schema(basic_function)
    assert result == {
        "type": "function",
        "function": {
            "name": "basic_function",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"},
                    "arg2": {"type": "string"},
                },
                "required": ["arg1", "arg2"],
            },
        },
    }

def test_complex_function():
    def complex_function_with_types_and_descriptions(
        arg1: int, arg2: str, arg3: float = 3.14, arg4: bool = False
    ):
        """This is a complex function with a docstring."""
        pass

    result = function_to_schema(complex_function_with_types_and_descriptions)
    assert result == {
        "type": "function",
        "function": {
            "name": "complex_function_with_types_and_descriptions",
            "description": "This is a complex function with a docstring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "integer"},
                    "arg2": {"type": "string"},
                    "arg3": {"type": "number"},
                    "arg4": {"type": "boolean"},
                },
                "required": ["arg1", "arg2"],
            },
        },
    }

# 测试function_to_schema的输出
def process_content(url: str, threhold: int = 1000) -> list:
    """
    从网页提取文本内容并将其处理成指定长度的段落。

    这个函数会获取指定URL的网页内容，提取其中的文本，
    并将文本重新组织成不超过指定长度阈值的段落。

    Args:
        url (str): 需要处理的网页URL
        threhold (int, optional): 每个段落的最大字符数。默认为1000。

    Returns:
        list: 处理后的段落列表，每个段落都是一个字符串，
             并以换行符结尾。

    Raises:
        requests.RequestException: 当网页请求失败时抛出
        BeautifulSoup.ParserError: 当HTML解析失败时抛出

    Examples:
        >>> url = "http://example.com"
        >>> paragraphs = process_content(url)
        >>> print(len(paragraphs))
        5
        >>> print(paragraphs[0][:50])
        'This is the beginning of the first paragraph...'

    Note:
        - 函数会自动处理超过阈值长度的段落，将其分割成多个较小的段落
        - 每个输出的段落都会以换行符结尾
        - 空白行会被自动过滤掉
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()

    # Reform the text into paragraphs
    candidate_paragraphs = (paragraph.strip() for paragraph in text.split("\n"))
    paragraphs = []
    current_paragraph = ""
    
    for candidate in candidate_paragraphs:
        if len(candidate) > 0:
            if len(current_paragraph) + len(candidate) <= threhold:
                current_paragraph += candidate + " "
            else:
                while len(candidate) > threhold:
                    if len(current_paragraph) > 0:
                        paragraphs.append(current_paragraph.strip() + "\n")
                        current_paragraph = ""
                    paragraphs.append(candidate[:threhold].strip() + "\n")
                    candidate = candidate[threhold:]
                if len(current_paragraph) + len(candidate) > threhold:
                    paragraphs.append(current_paragraph.strip() + "\n")
                    current_paragraph = candidate + " "
                else:
                    current_paragraph += candidate + " "
    
    if current_paragraph:
        paragraphs.append(current_paragraph.strip() + "\n")
    
    return paragraphs


# test_basic_function()
# test_complex_function()


# test
# schema = function_to_schema(process_content)
# print(json.dumps(schema, indent=4, ensure_ascii=False))
