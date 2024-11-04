"""
The assistant can execute Python code blocks.

It uses IPython to do so, and persists the IPython instance between calls to give a REPL-like experience.
"""

import base64
import dataclasses
import functools
import importlib.util
import logging
import re
import textwrap
import types
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    get_origin,
)

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from IPython.terminal.embed import InteractiveShellEmbed

# =============================================================================
# 基础设置
# =============================================================================
logger = getLogger(__name__)
console = Console(log_path=False)

# =============================================================================
# 从 message.py 和 base.py 提取的必要类和函数
# =============================================================================
@dataclass(frozen=True)
class Message:
    """A message in the assistant conversation."""
    role: Literal["system", "user", "assistant"]
    content: str
    pinned: bool = False
    hide: bool = False
    quiet: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    files: list[Path] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.timestamp, datetime)

@dataclass(frozen=True)
class Codeblock:
    """代码块类"""
    language: str
    content: str
    args: list[str]
    start: int | None = None

ConfirmFunc = Callable[[str], bool]

@dataclass(frozen=True, eq=False)
class ToolSpec:
    """Tool specification."""
    name: str
    desc: str
    instructions: str = ""
    examples: str = ""
    functions: list[Callable] | None = None
    init: Callable[[], Any] | None = None
    execute: Callable[[str, list[str], ConfirmFunc], Generator[Message, None, None]] | None = None
    block_types: list[str] = field(default_factory=list)
    available: bool = True

def print_preview(code: str, lang: str):
    """预览代码"""
    print("\n" + "="*80)
    print("代码预览")
    print("•"*80)
    syntax = Syntax(code.strip("\n"), lang, theme="monokai")
    console.print(syntax)
    print("="*80 + "\n")

# =============================================================================
# Python 工具实现
# =============================================================================
# IPython instance
_ipython: "InteractiveShellEmbed | None" = None
registered_functions: dict[str, Callable] = {}

T = TypeVar("T", bound=Callable)

def register_function(func: T) -> T:
    """Decorator to register a function to be available in the IPython instance."""
    registered_functions[func.__name__] = func
    if _ipython is not None:
        _ipython.push({func.__name__: func})
    return func

def derive_type(t) -> str:
    """推导类型字符串"""
    if get_origin(t) == Literal:
        v = ", ".join(f'"{a}"' for a in t.__args__)
        return f"Literal[{v}]"
    elif get_origin(t) == types.UnionType:
        v = ", ".join(derive_type(a) for a in t.__args__)
        return f"Union[{v}]"
    else:
        return t.__name__

def callable_signature(func: Callable) -> str:
    """获取函数签名字符串"""
    args = ", ".join(
        f"{k}: {derive_type(v)}"
        for k, v in func.__annotations__.items()
        if k != "return"
    )
    ret_type = func.__annotations__.get("return")
    ret = f" -> {derive_type(ret_type)}" if ret_type else ""
    return f"{func.__name__}({args}){ret}"

def get_functions_prompt() -> str:
    """获取可用函数的提示信息"""
    return "\n".join(
        f"- {callable_signature(func)}: {func.__doc__ or 'No description'}"
        for func in registered_functions.values()
    )

def _get_ipython():
    """获取或创建 IPython 实例"""
    global _ipython
    from IPython.terminal.embed import InteractiveShellEmbed

    if _ipython is None:
        _ipython = InteractiveShellEmbed()
        _ipython.push(registered_functions)

    return _ipython

def execute_python(
    code: str, args: list[str], confirm: ConfirmFunc = lambda _: True
) -> Generator[Message, None, None]:
    """执行 Python 代码并返回输出"""
    code = code.strip()
    print_preview(code, "python")
    if not confirm("Execute this code?"):
        yield Message("system", "Aborted, user chose not to run command.")
        return

    _ipython = _get_ipython()
    from IPython.utils.capture import capture_output

    with capture_output() as captured:
        result = _ipython.run_cell(code, silent=False, store_history=False)

    output = ""
    if isinstance(result.result, Message):
        yield result.result
        return

    if result.result is not None:
        output += f"Output: {result.result}\n"
    elif captured.stdout:
        output += f"Output: {captured.stdout.rstrip()}\n"

    if captured.stderr:
        output += f"Error: {captured.stderr.rstrip()}\n"
    if result.error_in_exec:
        tb = result.error_in_exec.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        output += f"Exception: {result.error_in_exec.__class__.__name__}: {result.error_in_exec}"

    output = re.sub(r"\x1b[^m]*m", "", output)
    yield Message("system", output.strip())

@functools.lru_cache
def get_installed_python_libraries() -> set[str]:
    """检查已安装的 Python 库"""
    candidates = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "sklearn",
        "statsmodels",
        "PIL",
    ]
    installed = set()
    for candidate in candidates:
        if importlib.util.find_spec(candidate):
            installed.add(candidate)
    return installed

# =============================================================================
# 工具配置
# =============================================================================
instructions = """
To execute Python code in an interactive IPython session, send a codeblock using the `ipython` language tag.
It will respond with the output and result of the execution.
If you first write the code in a normal python codeblock, remember to also execute it with the ipython codeblock.
"""

examples = """
#### Results of the last expression will be displayed, IPython-style:
> User: What is 2 + 2?
> Assistant: Let's compute this:
```ipython
2 + 2
```
> System: Executed code block.
Result: 4
"""

def init() -> ToolSpec:
    """初始化工具"""
    python_libraries = get_installed_python_libraries()
    python_libraries_str = "\n".join(f"- {lib}" for lib in python_libraries)

    _instructions = f"""{instructions}

The following libraries are available:
{python_libraries_str}

The following functions are available in the REPL:
{get_functions_prompt()}
    """.strip()

    return dataclasses.replace(tool, instructions=_instructions)

tool = ToolSpec(
    name="python",
    desc="Execute Python code",
    instructions=instructions,
    examples=examples,
    execute=execute_python,
    init=init,
    block_types=["ipython", "py"],
)

# =============================================================================
# 测试代码
# =============================================================================
class PythonToolTests:
    """Python工具测试类"""
    
    def __init__(self):
        self.tool = tool
        self.confirm = lambda _: True
        self.test_results = []  # 添加测试结果存储列表
    
    def record_test_result(self, case_name: str, case_type: str, 
                          expected: str, actual: str, passed: bool):
        """记录测试结果"""
        self.test_results.append({
            "案例名称": case_name,
            "案例类型": case_type,
            "期待结果": expected,
            "实际结果": actual,
            "是否通过": "✓" if passed else "✗"
        })
    
    def _run_test(self, test_type: str, code: str, expected: str):
        """通用测试运行器"""
        from rich.table import Table
        
        # 使用Panel和Text来创建居中的大字体标题
        print("\n")  # 空行分隔
        title = Text(f"测试类型: {test_type}", style="bold cyan", justify="center")
        title.stylize("bold magenta")
        panel = Panel(
            title,
            style="bold white",
            padding=(1, 0),
            width=80
        )
        console.print(panel)
        
        # 创建单个测试用例的表格
        table = Table(show_header=True, header_style="bold", border_style="bright_blue")
        table.add_column("项目", style="cyan", width=20)
        table.add_column("内容", style="yellow")
        
        # 添加测试代码到表格
        table.add_row("测试代码", code.strip())
        
        try:
            messages = list(execute_python(code.strip(), [], self.confirm))
            if not messages:
                raise AssertionError("没有产生输出")
                
            output = messages[0].content.strip()
            passed = expected in output
            self.record_test_result(
                code.strip().split('\n')[0][:30] + "...", 
                test_type,
                expected,
                output,
                passed
            )
            
            # 添加输出信息到表格
            table.add_row("实际输出", output)
            table.add_row("期望输出", expected)
            table.add_row(
                "测试结果", 
                "[green]✓ 通过[/green]" if passed else "[red]✗ 失败[/red]"
            )
            
            # 打印表格
            console.print(table)
            
            if not passed:
                raise AssertionError(f"输出不匹配\n期望: {expected}\n实际: {output}")
            
        except Exception as e:
            # 如果发生异常，添加错误信息到表格
            table.add_row("错误信息", str(e))
            table.add_row("测试结果", "[red]✗ 失败[/red]")
            console.print(table)
            raise

    def test_basic_execution(self):
        """测试基本的Python代码执行"""
        test_cases = [
            ("print('Hello, World!')", "Output: Hello, World!"),
            ("2 + 2", "Output: 4"),
            ("x = 10\nprint(x)", "Output: 10"),
            ("'test'.upper()", "Output: TEST"),
        ]
        
        for code, expected in test_cases:
            self._run_test("基本执行", code, expected)

    def test_error_handling(self):
        """测试错误处理"""
        error_cases = [
            ("print(undefined_variable)", "NameError"),
            ("1/0", "ZeroDivisionError"),
            ("invalid syntax", "SyntaxError"),
            ("import nonexistent_module", "ModuleNotFoundError"),
        ]
        
        for code, error_type in error_cases:
            self._run_test("错误处理", code, error_type)

    def test_multiline_code(self):
        """测试多行代码执行"""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

result = factorial(5)
print(f"5! = {result}")
"""
        self._run_test("多行代码", code, "5! = 120")

    def test_state_persistence(self):
        """测试状态持久化"""
        # 第一步：定义变量
        code1 = "test_var = 42"
        self._run_test("状态持久化-步骤1", code1, "")
        
        # 第二步：使用变量
        code2 = "print(test_var)"
        self._run_test("状态持久化-步骤2", code2, "42")

    def test_data_processing(self):
        """测试数据处理功能"""
        print("\n运行数据处理测试...")
        
        test_cases = [
            # 列表推导式测试
            ("""
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(squares)
            """, "Output: [1, 4, 9, 16, 25]"),
            
            # 字典操作测试
            ("""
data = {'a': 1, 'b': 2, 'c': 3}
data.update({'d': 4})
result = {k: v*2 for k, v in data.items()}
print(result)
            """, "Output: {'a': 2, 'b': 4, 'c': 6, 'd': 8}"),
            
            # 数据过滤测试
            ("""
numbers = range(-5, 6)
positives = list(filter(lambda x: x > 0, numbers))
print(positives)
            """, "Output: [1, 2, 3, 4, 5]"),
        ]
        
        for code, expected in test_cases:
            self._run_test("数据处理", code, expected)

    def test_advanced_features(self):
        """测试高级Python特性"""
        print("\n运行高级特性测试...")
        
        test_cases = [
            # 装饰器测试
            ("""
def logging_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logging_decorator
def greet(name):
    return f"Hello, {name}!"

result = greet("Python")
print(result)
            """, "Hello, Python!"),
            
            # 上下文管理器测试
            ("""
class Timer:
    def __enter__(self):
        print("Timer started")
        return self
    
    def __exit__(self, *args):
        print("Timer stopped")

with Timer():
    print("Operation in progress")
            """, "Operation in progress"),
            
            # 生成器测试
            ("""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

result = list(fibonacci(5))
print(result)
            """, "Output: [0, 1, 1, 2, 3]"),
        ]
        
        for code, expected in test_cases:
            self._run_test("高级特性", code, expected)

    def test_exception_handling(self):
        """测试异常处理"""
        print("\n运行异常处理测试...")
        
        test_cases = [
            # 自定义异常测试
            ("""
class CustomError(Exception):
    pass

try:
    raise CustomError("Custom error message")
except CustomError as e:
    print(f"Caught: {e}")
            """, "Caught: Custom error message"),
            
            # 多重异常处理
            ("""
try:
    result = 1 / 0
except (ZeroDivisionError, TypeError) as e:
    print(f"Handled error: {type(e).__name__}")
finally:
    print("Cleanup completed")
            """, "Cleanup completed"),
        ]
        
        for code, expected in test_cases:
            self._run_test("异常处理", code, expected)

    def test_class_features(self):
        """测试类相关特性"""
        print("\n运行类特性测试...")
        
        test_cases = [
            # 属性装饰器测试
            ("""
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

temp = Temperature(100)
print(f"Water boils at {temp.fahrenheit}°F")
            """, "Water boils at 212.0°F"),
            
            # 继承和多态测试
            ("""
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

animals = [Dog(), Cat()]
sounds = [animal.speak() for animal in animals]
print(sounds)
            """, "Output: ['Woof!', 'Meow!']"),
        ]
        
        for code, expected in test_cases:
            self._run_test("类特性", code, expected)

    def display_test_results(self):
        """以表格形式显示测试结果"""
        from rich.table import Table
        
        table = Table(title="Python工具测试结果汇总")
        
        # 添加表格列
        table.add_column("案例名称", style="cyan")
        table.add_column("案例类型", style="magenta")
        table.add_column("期待结果", style="green")
        table.add_column("实际结果", style="yellow")
        table.add_column("是否通过", style="bold")
        
        # 添加表格行
        for result in self.test_results:
            table.add_row(
                result["案例名称"],
                result["案例类型"],
                result["期待结果"],
                result["实际结果"],
                result["是否通过"]
            )
        
        # 打印表格
        console.print("\n")
        console.print(table)
        
        # 打印统计信息
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["是否通过"] == "✓")
        print(f"\n测试统计:")
        print(f"总计测试: {total}")
        print(f"通过测试: {passed}")
        print(f"失败测试: {total - passed}")
        
        if passed == total:
            print("\n✨ 所有测试通过！")
        else:
            print("\n❌ 存在失败的测试")

    def run_all_tests(self):
        """运行所有测试"""
        print("开始运行Python工具测试...\n")
        
        tests = [
            self.test_basic_execution,
            self.test_error_handling,
            self.test_multiline_code,
            self.test_state_persistence,
            self.test_data_processing,      # 新增
            self.test_advanced_features,    # 新增
            self.test_exception_handling,   # 新增
            self.test_class_features,       # 新增
        ]
        
        failed_tests = []
        for test in tests:
            try:
                test()
            except Exception as e:
                failed_tests.append((test.__name__, str(e)))
        
        # 显示测试结果表格
        self.display_test_results()
        
        # 如果有失败的测试，显示详细错误信息
        if failed_tests:
            print("\n详细错误信息:")
            for test_name, error in failed_tests:
                print(f"- {test_name}: {error}")

if __name__ == "__main__":
    tests = PythonToolTests()
    tests.run_all_tests()