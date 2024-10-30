from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from inspect import signature

"""
Python函数的签名信息主要包含以下几个部分：

1、函数名称 (Function Name)
用于标识和调用函数的名称


2、参数列表 (Parameters)：
必需参数（Required Parameters）: 调用时必须提供的参数
默认参数（Default Parameters）: 有默认值的参数
可选参数（Optional Parameters）: 使用 Optional[type] 标注的可选参数
可变位置参数（*args）: 接收任意数量的位置参数
可变关键字参数（**kwargs）: 接收任意数量的关键字参数


3、类型注解 (Type Hints)：
参数类型注解：指定参数的预期类型
返回值类型注解：指定函数返回值的类型
可以使用 typing 模块中的类型


4、返回值声明 (Return Annotation)：
使用 -> 符号后跟返回类型
可以是具体类型、Union类型或None


5、装饰器 (Decorators)：
可选的函数修饰器，如 @dataclass, @staticmethod 等

"""

# 基本函数签名
def basic_function(name: str, age: int = 10) -> bool:
    return True
 
# 复杂函数签名示例
def complex_function(
    required_param: str,                     # 必需参数
    *args: tuple,                           # 可变位置参数
    default_param: int = 100,               # 默认参数
    optional_param: Optional[str] = None,    # 可选参数
    **kwargs: dict                          # 可变关键字参数
) -> Union[List[str], Dict[str, Any]]:      # 联合返回类型
    return []

# 使用dataclass的函数签名
@dataclass
class UserInfo:
    name: str
    age: int
    
def process_user(user: UserInfo) -> None:
    pass

# 获取函数签名信息
def print_signature_info(func):
    sig = signature(func)
    print(f"函数名: {func.__name__}")
    print(f"参数: {sig.parameters}")
    print(f"返回注解: {sig.return_annotation}")
    print("---")

# 打印各个函数的签名信息
print_signature_info(basic_function)
print_signature_info(complex_function)
print_signature_info(process_user)
