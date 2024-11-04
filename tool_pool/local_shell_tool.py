"""
The assistant can execute shell commands with bash by outputting code blocks with `shell` as the language.
"""

# Standard library imports
import atexit
import base64
import dataclasses
import functools
import io
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

# Third-party imports
import bashlex
from lxml import etree
from rich.console import Console
from rich.syntax import Syntax
import tiktoken
import tomlkit
from tomlkit._utils import escape_string
from typing_extensions import Self

# Local imports
# (如果有本地导入，放在这里)


mode: Literal["markdown", "xml"] = "markdown"
console = Console(log_path=False)
ConfirmFunc = Callable[[str], bool]
logger = logging.getLogger(__name__)
exclusive_mode = False
InitFunc: TypeAlias = Callable[[], "ToolSpec"]


max_system_len = 20000
# available providers
Provider = Literal[
    "openai", "anthropic", "azure", "openrouter", "groq", "xai", "deepseek", "local"
]

ProvidersWithFiles: list[Provider] = ["openai", "anthropic", "openrouter"]



def print_preview(code: str, lang: str):  # pragma: no cover
    print()
    print("[bold white]Preview[/bold white]")
    # NOTE: we can set background_color="default" to remove background
    print(Syntax(code.strip("\n"), lang))
    print()



ROLE_COLOR = {
    "user": "green",
    "assistant": "green",
    "system": "grey42",
}



def get_tokenizer(model: str):
    if "gpt-4" in model or "gpt-3.5" in model:
        return tiktoken.encoding_for_model(model)
    else:  # pragma: no cover
        logger.warning(
            f"No encoder implemented for model {model}."
            "Defaulting to tiktoken cl100k_base encoder."
            "Use results only as estimates."
        )
        return tiktoken.get_encoding("cl100k_base")





@dataclass(frozen=True)
class Codeblock:
    lang: str
    content: str
    path: str | None = None
    start: int | None = field(default=None, compare=False)

    def __post_init__(self):
        # init path if path is None and lang is pathy
        if self.path is None and self.is_filename:
            object.__setattr__(self, "path", self.lang)  # frozen dataclass workaround

    def to_markdown(self) -> str:
        return f"```{self.lang}\n{self.content}\n```"

    def to_xml(self) -> str:
        return f'<codeblock lang="{self.lang}" path="{self.path}">\n{self.content}\n</codeblock>'

    @classmethod
    def from_markdown(cls, content: str) -> "Codeblock":
        if content.strip().startswith("```"):
            content = content[3:]
        if content.strip().endswith("```"):
            content = content[:-3]
        lang = content.splitlines()[0].strip()
        return cls(lang, content[len(lang) :])

    @classmethod
    def from_xml(cls, content: str) -> "Codeblock":
        """
        Example:
          <codeblock lang="python" path="example.py">
          print("Hello, world!")
          </codeblock>
        """
        root = ElementTree.fromstring(content)
        return cls(root.attrib["lang"], root.text or "", root.attrib.get("path"))

    @property
    def is_filename(self) -> bool:
        return "." in self.lang or "/" in self.lang

    @classmethod
    def iter_from_markdown(cls, markdown: str) -> list["Codeblock"]:
        return list(_extract_codeblocks(markdown))


@dataclass(frozen=True, eq=False)
class Message:
    """
    A message in the assistant conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The content of the message.
        pinned: Whether this message should be pinned to the top of the chat, and never context-trimmed.
        hide: Whether this message should be hidden from the chat output (but still be sent to the assistant).
        quiet: Whether this message should be printed on execution (will still print on resume, unlike hide).
               This is not persisted to the log file.
        timestamp: The timestamp of the message.
        files: Files attached to the message, could e.g. be images for vision.
    """

    role: Literal["system", "user", "assistant"]
    content: str
    pinned: bool = False
    hide: bool = False
    quiet: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    files: list[Path] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.timestamp, datetime)
        if self.role == "system":
            if (length := len_tokens(self)) >= max_system_len:
                logger.warning(f"System message too long: {length} tokens")

    def __repr__(self):
        content = textwrap.shorten(self.content, 20, placeholder="...")
        return f"<Message role={self.role} content={content}>"

    def __eq__(self, other):
        # FIXME: really include timestamp?
        if not isinstance(other, Message):
            return False
        return (
            self.role == other.role
            and self.content == other.content
            and self.timestamp == other.timestamp
        )

    def replace(self, **kwargs) -> Self:
        """Replace attributes of the message."""
        return dataclasses.replace(self, **kwargs)

    def _content_files_list(
        self,
        provider: Provider,
    ) -> list[dict[str, Any]]:
        # only these providers support files in the content
        if provider not in ProvidersWithFiles:
            raise ValueError("Provider does not support files in the content")

        # combines a content message with a list of files
        content: list[dict[str, Any]] = (
            self.content
            if isinstance(self.content, list)
            else [{"type": "text", "text": self.content}]
        )
        allowed_file_exts = ["jpg", "jpeg", "png", "gif"]

        for f in self.files:
            ext = f.suffix[1:]
            if ext not in allowed_file_exts:
                logger.warning("Unsupported file type: %s", ext)
                continue
            if ext == "jpg":
                ext = "jpeg"
            media_type = f"image/{ext}"
            content.append(
                {
                    "type": "text",
                    "text": f"![{f.name}]({f.name}):",
                }
            )

            # read file
            data_bytes = f.read_bytes()
            data = base64.b64encode(data_bytes).decode("utf-8")

            # check that the file is not too large
            # anthropic limit is 5MB, seems to measure the base64-encoded size instead of raw bytes
            # TODO: use compression to reduce file size
            # print(f"{len(data)=}")
            if len(data) > 5_000_000:
                content.append(
                    {
                        "type": "text",
                        "text": "Image size exceeds 5MB. Please upload a smaller image.",
                    }
                )
                continue

            if provider == "anthropic":
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            elif provider == "openai":
                # OpenAI format
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
            else:
                # Storage/wire format (keep files in `files` list)
                # Do nothing to integrate files into the message content
                pass

        return content

    def to_dict(self, keys=None, provider: Provider | None = None) -> dict:
        """Return a dict representation of the message, serializable to JSON."""
        content: str | list[dict[str, Any]]
        if provider in ProvidersWithFiles:
            # OpenAI/Anthropic format should include files in the content
            # Some otherwise OpenAI-compatible providers (groq, deepseek?) do not support this
            content = self._content_files_list(provider)
        else:
            # storage/wire format should keep the content as a string
            content = self.content

        d: dict = {
            "role": self.role,
            "content": content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.files:
            d["files"] = [str(f) for f in self.files]
        if self.pinned:
            d["pinned"] = True
        if self.hide:
            d["hide"] = True
        if keys:
            return {k: d[k] for k in keys}
        return d

    def to_xml(self) -> str:
        """Converts a message to an XML string."""
        attrs = f"role='{self.role}'"
        return f"<message {attrs}>\n{self.content}\n</message>"

    def format(self, oneline: bool = False, highlight: bool = False) -> str:
        return format_msgs([self], oneline=oneline, highlight=highlight)[0]

    def print(self, oneline: bool = False, highlight: bool = True) -> None:
        print_msg(self, oneline=oneline, highlight=highlight)

    def to_toml(self) -> str:
        """Converts a message to a TOML string, for easy editing by hand in editor to then be parsed back."""
        flags = []
        if self.pinned:
            flags.append("pinned")
        if self.hide:
            flags.append("hide")
        flags_toml = "\n".join(f"{flag} = true" for flag in flags)
        files_toml = f"files = {[str(f) for f in self.files]}" if self.files else ""
        extra = (flags_toml + "\n" + files_toml).strip()

        # doublequotes need to be escaped
        # content = self.content.replace('"', '\\"')
        content = escape_string(self.content)
        content = content.replace("\\n", "\n")
        content = content.strip()

        return f'''[message]
role = "{self.role}"
content = """
{content}
"""
timestamp = "{self.timestamp.isoformat()}"
{extra}
'''

    @classmethod
    def from_toml(cls, toml: str) -> Self:
        """
        Converts a TOML string to a message.

        The string can be a single [[message]].
        """

        t = tomlkit.parse(toml)
        assert "message" in t and isinstance(t["message"], dict)
        msg: dict = t["message"]  # type: ignore

        return cls(
            msg["role"],
            msg["content"].strip(),
            pinned=msg.get("pinned", False),
            hide=msg.get("hide", False),
            files=[Path(f) for f in msg.get("files", [])],
            timestamp=datetime.fromisoformat(msg["timestamp"]),
        )

    def get_codeblocks(self) -> list[Codeblock]:
        """
        Get all codeblocks from the message content.
        """
        content_str = self.content

        # prepend newline to make sure we get the first codeblock
        if not content_str.startswith("\n"):
            content_str = "\n" + content_str

        # check if message contains a code block
        backtick_count = content_str.count("\n```")
        if backtick_count < 2:
            return []

        return Codeblock.iter_from_markdown(content_str)


class ExecuteFunc(Protocol):
    def __call__(
        self, code: str, args: list[str], confirm: ConfirmFunc
    ) -> Generator[Message, None, None]: ...

def transform_examples_to_chat_directives(examples: str) -> str:
    """Transform examples into chat directives format."""
    if not examples:
        return ""
    
    # Split examples into sections by ####
    sections = examples.split("####")
    
    transformed = []
    for section in sections:
        if not section.strip():
            continue
            
        # Add chat directive format
        lines = section.strip().split("\n")
        transformed.append(".. chat::")
        transformed.append("    :hide-output:")
        transformed.append("")
        
        # Indent the content
        for line in lines:
            transformed.append(f"    {line}")
        transformed.append("")
        
    return "\n".join(transformed)


@dataclass(frozen=True, eq=False)
class ToolSpec:
    """
    Tool specification. Defines a tool that can be used by the agent.

    Args:
        name: The name of the tool.
        desc: A description of the tool.
        instructions: Instructions on how to use the tool.
        examples: Example usage of the tool.
        functions: Functions registered in the IPython REPL.
        init: An optional function that is called when the tool is first loaded.
        execute: An optional function that is called when the tool executes a block.
        block_types: A list of block types that the tool will execute.
        available: Whether the tool is available for use.
    """

    name: str
    desc: str
    instructions: str = ""
    examples: str = ""
    functions: list[Callable] | None = None
    init: InitFunc | None = None
    execute: ExecuteFunc | None = None
    block_types: list[str] = field(default_factory=list)
    available: bool = True

    def get_doc(self, doc: str | None = None) -> str:
        """Returns an updated docstring with examples."""
        if not doc:
            doc = ""
        else:
            doc += "\n\n"
        if self.instructions:
            doc += f"""
.. rubric:: Instructions

.. code-block:: markdown

{indent(self.instructions, "    ")}\n\n"""
        if self.examples:
            doc += f"""
.. rubric:: Examples

{transform_examples_to_chat_directives(self.examples)}\n\n
"""
        # doc += """.. rubric:: Members"""
        return doc.strip()

    def __eq__(self, other):
        if not isinstance(other, ToolSpec):
            return False
        return self.name == other.name


@dataclass(frozen=True)
class ToolUse:
    tool: str
    args: list[str]
    content: str
    start: int | None = None

    def execute(self, confirm: ConfirmFunc) -> Generator[Message, None, None]:
        """Executes a tool-use tag and returns the output."""
        # noreorder
        from . import get_tool  # fmt: skip

        tool = get_tool(self.tool)
        if tool and tool.execute:
            try:
                yield from tool.execute(self.content, self.args, confirm)
            except Exception as e:
                # if we are testing, raise the exception
                if "pytest" in globals():
                    raise e
                yield Message("system", f"Error executing tool '{self.tool}': {e}")
        else:
            logger.warning(f"Tool '{self.tool}' is not available for execution.")

    @property
    def is_runnable(self) -> bool:
        # noreorder
        from . import get_tool  # fmt: skip

        tool = get_tool(self.tool)
        return bool(tool.execute) if tool else False

    @classmethod
    def _from_codeblock(cls, codeblock: Codeblock) -> "ToolUse | None":
        """Parses a codeblock into a ToolUse. Codeblock must be a supported type.

        Example:
          ```lang
          content
          ```
        """
        # noreorder
        from . import get_tool_for_langtag  # fmt: skip

        if tool := get_tool_for_langtag(codeblock.lang):
            # NOTE: special case
            args = (
                codeblock.lang.split(" ")[1:]
                if tool.name != "save"
                else [codeblock.lang]
            )
            return ToolUse(tool.name, args, codeblock.content, start=codeblock.start)
        else:
            # no_op_langs = ["csv", "json", "html", "xml", "stdout", "stderr", "result"]
            # if codeblock.lang and codeblock.lang not in no_op_langs:
            #     logger.warning(
            #         f"Unknown codeblock type '{codeblock.lang}', neither supported language or filename."
            #     )
            return None

    @classmethod
    def iter_from_content(cls, content: str) -> Generator["ToolUse", None, None]:
        """Returns all ToolUse in a message, markdown or XML, in order."""
        # collect all tool uses
        tool_uses = []
        if mode == "xml" or not exclusive_mode:
            for tool_use in cls._iter_from_xml(content):
                tool_uses.append(tool_use)
        if mode == "markdown" or not exclusive_mode:
            for tool_use in cls._iter_from_markdown(content):
                tool_uses.append(tool_use)

        # return them in the order they appear
        assert all(x.start is not None for x in tool_uses)
        tool_uses.sort(key=lambda x: x.start or 0)
        for tool_use in tool_uses:
            yield tool_use

    @classmethod
    def _iter_from_markdown(cls, content: str) -> Generator["ToolUse", None, None]:
        """Returns all markdown-style ToolUse in a message.

        Example:
          ```ipython
          print("Hello, world!")
          ```
        """
        for codeblock in Codeblock.iter_from_markdown(content):
            if tool_use := cls._from_codeblock(codeblock):
                yield tool_use

    @classmethod
    def _iter_from_xml(cls, content: str) -> Generator["ToolUse", None, None]:
        """Returns all XML-style ToolUse in a message.

        Example:
          <tool-use>
          <ipython>
          print("Hello, world!")
          </ipython>
          </tool-use>
        """
        if "<tool-use>" not in content:
            return
        if "</tool-use>" not in content:
            return

        try:
            # Parse the content as HTML to be more lenient with malformed XML
            parser = etree.HTMLParser()
            tree = etree.fromstring(content, parser)

            for tooluse in tree.xpath("//tool-use"):
                for child in tooluse.getchildren():
                    tool_name = child.tag
                    args = list(child.attrib.values())
                    tool_content = (child.text or "").strip()

                    # Find the start position of the tool in the original content
                    start_pos = content.find(f"<{tool_name}")

                    yield ToolUse(
                        tool_name,
                        args,
                        tool_content,
                        start=start_pos,
                    )
        except etree.ParseError as e:
            logger.warning(f"Failed to parse XML content: {e}")
            return

    def to_output(self) -> str:
        if mode == "markdown":
            return self._to_markdown()
        elif mode == "xml":
            return self._to_xml()

    def _to_markdown(self) -> str:
        args = " ".join(self.args)
        return f"```{self.tool} {args}\n{self.content}\n```"

    def _to_xml(self) -> str:
        args = " ".join(self.args)
        args_str = "" if not args else f" args='{args}'"
        return f"<tool-use>\n<{self.tool}{args_str}>\n{self.content}\n</{self.tool}>\n</tool-use>"



def rich_to_str(s: Any, **kwargs) -> str:
    c = Console(file=io.StringIO(), **kwargs)
    c.print(s)
    return c.file.getvalue()  # type: ignore



def _extract_codeblocks(markdown: str) -> Generator[Codeblock, None, None]:
    # speed check (early exit): check if message contains a code block
    backtick_count = markdown.count("```")
    if backtick_count < 2:
        return

    lines = markdown.split("\n")
    stack: list[str] = []
    current_block = []
    current_lang = ""

    for idx, line in enumerate(lines):
        # not actually the starting index, but close enough
        # TODO: fix to actually be correct
        start_idx = sum(len(line) + 1 for line in lines[:idx])
        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            if not stack:  # Start of a new block
                stack.append(stripped_line[3:])
                current_lang = stripped_line[3:]
            elif stripped_line[3:] and stack[-1] != stripped_line[3:]:  # Nested start
                current_block.append(line)
                stack.append(stripped_line[3:])
            else:  # End of a block
                if len(stack) == 1:  # Outermost block
                    yield Codeblock(
                        current_lang, "\n".join(current_block), start=start_idx
                    )
                    current_block = []
                    current_lang = ""
                else:  # Nested end
                    current_block.append(line)
                stack.pop()
        elif stack:
            current_block.append(line)



# TODO: remove model assumption
def len_tokens(content: str | Message | list[Message], model: str = "gpt-4") -> int:
    """Get the number of tokens in a string, message, or list of messages."""
    if isinstance(content, list):
        return sum(len_tokens(msg.content, model) for msg in content)
    if isinstance(content, Message):
        return len_tokens(content.content, model)
    return len(get_tokenizer(model).encode(content))



def format_msgs(
    msgs: list[Message],
    oneline: bool = False,
    highlight: bool = False,
    indent: int = 0,
) -> list[str]:
    """Formats messages for printing to the console."""
    outputs = []
    for msg in msgs:
        userprefix = msg.role.capitalize()
        if highlight:
            color = ROLE_COLOR[msg.role]
            userprefix = f"[bold {color}]{userprefix}[/bold {color}]"
        # get terminal width
        max_len = shutil.get_terminal_size().columns - len(userprefix)
        output = ""
        if oneline:
            output += textwrap.shorten(
                msg.content.replace("\n", "\\n"), width=max_len, placeholder="..."
            )
            if len(output) < 20:
                output = msg.content.replace("\n", "\\n")[:max_len] + "..."
        else:
            multiline = len(msg.content.split("\n")) > 1
            output += "\n" + indent * " " if multiline else ""
            for i, block in enumerate(msg.content.split("```")):
                if i % 2 == 0:
                    output += textwrap.indent(block, prefix=indent * " ")
                    continue
                elif highlight:
                    lang = block.split("\n")[0]
                    block = rich_to_str(Syntax(block.rstrip(), lang))
                output += f"```{block.rstrip()}\n```"
        outputs.append(f"{userprefix}: {output.rstrip()}")
    return outputs


def print_msg(
    msg: Message | list[Message],
    oneline: bool = False,
    highlight: bool = True,
    show_hidden: bool = False,
) -> None:
    """Prints the log to the console."""
    # if not tty, force highlight=False (for tests and such)
    if not sys.stdout.isatty():
        highlight = False

    msgs = msg if isinstance(msg, list) else [msg]
    msgstrs = format_msgs(msgs, highlight=highlight, oneline=oneline)
    skipped_hidden = 0
    for m, s in zip(msgs, msgstrs):
        if m.hide and not show_hidden:
            skipped_hidden += 1
            continue
        try:
            console.print(s)
        except Exception:
            # rich can throw errors, if so then print the raw message
            logger.exception("Error printing message")
            print(s)
    if skipped_hidden:
        console.print(
            f"[grey30]Skipped {skipped_hidden} hidden system messages, show with --show-hidden[/]"
        )




@functools.lru_cache
def get_installed_programs() -> set[str]:
    candidates = [
        # platform-specific
        "brew",
        "apt-get",
        "pacman",
        # common and useful
        "ffmpeg",
        "magick",
        "pandoc",
        "git",
        "docker",
    ]
    installed = set()
    for candidate in candidates:
        if shutil.which(candidate) is not None:
            installed.add(candidate)
    return installed


shell_programs_str = "\n".join(f"- {prog}" for prog in get_installed_programs())
is_macos = sys.platform == "darwin"

instructions = f"""
When you send a message containing bash code, it will be executed in a stateful bash shell.
The shell will respond with the output of the execution.
Do not use EOF/HereDoc syntax to send multiline commands, as the assistant will not be able to handle it.

These programs are available, among others:
{shell_programs_str}
""".strip()

examples = f"""
User: list the current directory
Assistant: To list the files in the current directory, use `ls`:
{ToolUse("shell", [], "ls").to_output()}
System: Ran command: `ls`
{ToolUse("shell", [], '''
file1.txt
file2.txt
'''.strip()).to_output()}

#### The assistant can learn context by exploring the filesystem
User: learn about the project
Assistant: Lets start by checking the files
{ToolUse("shell", [], "git ls-files").to_output()}
System:
{ToolUse("stdout", [], '''
README.md
main.py
'''.strip()).to_output()}
Assistant: Now lets check the README
{ToolUse("shell", [], "cat README.md").to_output()}
System:
{ToolUse("stdout", [], "(contents of README.md)").to_output()}
Assistant: Now we check main.py
{ToolUse("shell", [], "cat main.py").to_output()}
System:
{ToolUse("stdout", [], "(contents of main.py)").to_output()}
Assistant: The project is...


#### Create vue project
User: Create a new vue project with typescript and pinia named fancy-project
Assistant: Sure! Let's create a new vue project with TypeScript and Pinia named fancy-project:
{ToolUse("shell", [], "npm init vue@latest fancy-project --yes -- --typescript --pinia").to_output()}
System:
{ToolUse("stdout", [], '''
> npx
> create-vue

Vue.js - The Progressive JavaScript Framework

Scaffolding project in ./fancy-project...
'''.strip()).to_output()}
"""


class ShellSession:
    process: subprocess.Popen
    stdout_fd: int
    stderr_fd: int
    delimiter: str

    def __init__(self) -> None:
        self._init()

        # close on exit
        atexit.register(self.close)

    def _init(self):
        self.process = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
            universal_newlines=True,
        )
        self.stdout_fd = self.process.stdout.fileno()  # type: ignore
        self.stderr_fd = self.process.stderr.fileno()  # type: ignore
        self.delimiter = "END_OF_COMMAND_OUTPUT"

        # set GIT_PAGER=cat
        self.run("export GIT_PAGER=cat")

    def run(self, code: str, output=True) -> tuple[int | None, str, str]:
        """Runs a command in the shell and returns the output."""
        commands = split_commands(code)
        res_code: int | None = None
        res_stdout, res_stderr = "", ""
        for cmd in commands:
            res_cur = self._run(cmd, output=output)
            res_code = res_cur[0]
            res_stdout += res_cur[1]
            res_stderr += res_cur[2]
            if res_code != 0:
                return res_code, res_stdout, res_stderr
        return res_code, res_stdout, res_stderr

    def _run(self, command: str, output=True, tries=0) -> tuple[int | None, str, str]:
        assert self.process.stdin

        # run the command
        full_command = f"{command}; echo ReturnCode:$? {self.delimiter}\n"
        try:
            self.process.stdin.write(full_command)
        except BrokenPipeError:
            # process has died
            if tries == 0:
                # log warning and restart, once
                logger.warning("Warning: shell process died, restarting")
                self.restart()
                return self._run(command, output=output, tries=tries + 1)
            else:
                raise

        self.process.stdin.flush()

        stdout = []
        stderr = []
        return_code = None
        read_delimiter = False

        while True:
            rlist, _, _ = select.select([self.stdout_fd, self.stderr_fd], [], [])
            for fd in rlist:
                assert fd in [self.stdout_fd, self.stderr_fd]
                # We use a higher value, because there is a bug which leads to spaces at the boundary
                # 2**12 = 4096
                # 2**16 = 65536
                data = os.read(fd, 2**16).decode("utf-8")
                for line in re.split(r"(\n)", data):
                    if "ReturnCode:" in line:
                        return_code_str = (
                            line.split("ReturnCode:")[1].split(" ")[0].strip()
                        )
                        return_code = int(return_code_str)
                    if self.delimiter in line:
                        read_delimiter = True
                        continue
                    if fd == self.stdout_fd:
                        stdout.append(line)
                        if output:
                            print(line, end="", file=sys.stdout)
                    elif fd == self.stderr_fd:
                        stderr.append(line)
                        if output:
                            print(line, end="", file=sys.stderr)
            if read_delimiter:
                break

        # if command is cd and successful, we need to change the directory
        if command.startswith("cd ") and return_code == 0:
            ex, pwd, _ = self._run("pwd", output=False)
            assert ex == 0
            os.chdir(pwd.strip())

        return (
            return_code,
            "".join(stdout).replace(f"ReturnCode:{return_code}", "").strip(),
            "".join(stderr).strip(),
        )

    def close(self):
        assert self.process.stdin
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait(timeout=0.2)
        self.process.kill()

    def restart(self):
        self.close()
        self._init()


_shell: ShellSession | None = None


def get_shell() -> ShellSession:
    global _shell
    if _shell is None:
        # init shell
        _shell = ShellSession()
    return _shell


# used in testing
def set_shell(shell: ShellSession) -> None:
    global _shell
    _shell = shell


# NOTE: This does not handle control flow words like if, for, while.
cmd_regex = re.compile(r"(?:^|[|&;]|\|\||&&|\n)\s*([^\s|&;]+)")


def execute_shell(
    code: str, args: list[str], confirm: ConfirmFunc
) -> Generator[Message, None, None]:
    """Executes a shell command and returns the output."""
    shell = get_shell()
    assert not args
    allowlist_commands = ["ls", "stat", "cd", "cat", "pwd", "echo", "head"]
    allowlisted = True

    cmd = code.strip()
    if cmd.startswith("$ "):
        cmd = cmd[len("$ ") :]

    for match in cmd_regex.finditer(cmd):
        for group in match.groups():
            if group and group not in allowlist_commands:
                allowlisted = False
                break

    if not allowlisted:
        print_preview(cmd, "bash")
        if not confirm("Run command?"):
            yield Message("system", "User chose not to run command.")
            return

    try:
        returncode, stdout, stderr = shell.run(cmd)
    except Exception as e:
        yield Message("system", f"Error: {e}")
        return
    stdout = _shorten_stdout(stdout.strip(), pre_tokens=2000, post_tokens=8000)
    stderr = _shorten_stdout(stderr.strip(), pre_tokens=2000, post_tokens=2000)

    msg = (
        _format_block_smart(
            f"Ran {'allowlisted ' if allowlisted else ''}command", cmd, lang="bash"
        )
        + "\n\n"
    )
    if stdout:
        msg += _format_block_smart("", stdout, "stdout") + "\n\n"
    if stderr:
        msg += _format_block_smart("", stderr, "stderr") + "\n\n"
    if not stdout and not stderr:
        msg += "No output\n"
    if returncode:
        msg += f"Return code: {returncode}"

    yield Message("system", msg)


def _format_block_smart(header: str, cmd: str, lang="") -> str:
    # prints block as a single line if it fits, otherwise as a code block
    s = ""
    if header:
        s += f"{header}:"
    if len(cmd.split("\n")) == 1:
        s += f" `{cmd}`"
    else:
        s += f"\n```{lang}\n{cmd}\n```"
    return s


def _shorten_stdout(
    stdout: str,
    pre_lines=None,
    post_lines=None,
    pre_tokens=None,
    post_tokens=None,
    strip_dates=False,
    strip_common_prefix_lines=0,
) -> str:
    lines = stdout.split("\n")

    # NOTE: This can cause issues when, for example, reading a CSV with dates in the first column
    if strip_dates:
        # strip iso8601 timestamps
        lines = [
            re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.]\d{3,9}Z?", "", line)
            for line in lines
        ]
        # strip dates like "2017-08-02 08:48:43 +0000 UTC"
        lines = [
            re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}( [+]\d{4})?( UTC)?", "", line)
            for line in lines
        ]

    # strip common prefixes, useful for things like `gh runs view`
    if strip_common_prefix_lines and len(lines) >= strip_common_prefix_lines:
        prefix = os.path.commonprefix([line.rstrip() for line in lines])
        if prefix:
            lines = [line[len(prefix) :] for line in lines]

    # check that if pre_lines is set, so is post_lines, and vice versa
    assert (pre_lines is None) == (post_lines is None)
    if (
        pre_lines is not None
        and post_lines is not None
        and len(lines) > pre_lines + post_lines
    ):
        lines = (
            lines[:pre_lines]
            + [f"... ({len(lines) - pre_lines - post_lines} truncated) ..."]
            + lines[-post_lines:]
        )

    # check that if pre_tokens is set, so is post_tokens, and vice versa
    assert (pre_tokens is None) == (post_tokens is None)
    if pre_tokens is not None and post_tokens is not None:
        tokenizer = get_tokenizer("gpt-4")  # TODO: use sane default
        tokens = tokenizer.encode(stdout)
        if len(tokens) > pre_tokens + post_tokens:
            lines = (
                [tokenizer.decode(tokens[:pre_tokens])]
                + ["... (truncated output) ..."]
                + [tokenizer.decode(tokens[-post_tokens:])]
            )

    return "\n".join(lines)


def split_commands(script: str) -> list[str]:
    # TODO: write proper tests
    parts = bashlex.parse(script)
    commands = []
    for part in parts:
        if part.kind == "command":
            command_parts = []
            for word in part.parts:
                start, end = word.pos
                command_parts.append(script[start:end])
            command = " ".join(command_parts)
            commands.append(command)
        elif part.kind == "compound":
            for node in part.list:
                command_parts = []
                for word in node.parts:
                    start, end = word.pos
                    command_parts.append(script[start:end])
                command = " ".join(command_parts)
                commands.append(command)
        elif part.kind in ["function", "pipeline", "list"]:
            commands.append(script[part.pos[0] : part.pos[1]])
        else:
            logger.warning(
                f"Unknown shell script part of kind '{part.kind}', hoping this works"
            )
            commands.append(script[part.pos[0] : part.pos[1]])
    return commands


tool = ToolSpec(
    name="shell",
    desc="Executes shell commands.",
    instructions=instructions,
    examples=examples,
    execute=execute_shell,
    block_types=["shell"],
)
__doc__ = tool.get_doc(__doc__)

# 在文件末尾添加测试代码
def test_shell_basic_commands():
    """测试基本shell命令执行"""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            shell = ShellSession()
            try:
                # 测试 echo 命令
                code, stdout, stderr = shell.run('echo "hello world"')
                assert code == 0, "Echo command failed"
                assert stdout.strip() == "hello world", "Echo output incorrect"
                assert stderr == "", "Echo produced error output"

                # 测试 pwd 命令
                code, stdout, stderr = shell.run('pwd')
                assert code == 0, "PWD command failed"
                assert os.path.realpath(stdout.strip()) == os.path.realpath(temp_dir), "PWD output incorrect"
                assert stderr == "", "PWD produced error output"
                print("✓ Basic commands test passed")
            finally:
                shell.close()
        finally:
            os.chdir(old_cwd)

def test_shell_cd_command():
    """测试 cd 命令是否正确改变目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            shell = ShellSession()
            try:
                # 创建测试目录
                test_dir = "test_dir"
                os.mkdir(test_dir)
                
                # 执行 cd 命令
                code, stdout, stderr = shell.run(f'cd {test_dir}')
                assert code == 0, "CD command failed"
                
                # 验证目录是否改变
                code, stdout, stderr = shell.run('pwd')
                assert stdout.strip().endswith(test_dir), "Directory change failed"
                print("✓ CD command test passed")
            finally:
                shell.close()
        finally:
            os.chdir(old_cwd)

def test_split_commands():
    """测试命令分割功能"""
    # 测试简单命令分割
    commands = split_commands('echo "hello"; ls')
    assert len(commands) == 1, "Command split failed"
    assert 'echo "hello"; ls' in commands, "Commands missing"

    # 测试管道命令
    commands = split_commands('ls | grep test')
    assert len(commands) == 1, "Pipe command split failed"
    assert 'ls | grep test' == commands[0], "Pipe command incorrect"

    # 测试 && 操作符
    commands = split_commands('echo "hello" && ls')
    assert len(commands) == 1, "AND operator command split failed"
    assert 'echo "hello" && ls' == commands[0], "AND operator command incorrect"
    
    print("✓ Split commands test passed")

def test_shorten_stdout():
    """测试输出截断功能"""
    # 测试行数限制
    long_output = "\n".join(str(i) for i in range(100))
    shortened = _shorten_stdout(long_output, pre_lines=5, post_lines=5)
    assert "truncated" in shortened, "Output not truncated"
    assert shortened.count("\n") < 100, "Output not shortened"

    # 测试日期去除
    date_output = "2023-01-01T12:00:00.000Z some text"
    cleaned = _shorten_stdout(date_output, strip_dates=True)
    assert "2023-01-01" not in cleaned, "Date not removed"
    print("✓ Shorten stdout test passed")

def test_get_installed_programs():
    """测试已安装程序检测"""
    programs = get_installed_programs()
    assert isinstance(programs, set), "Programs not returned as set"
    assert any(cmd in programs for cmd in ['git', 'ls', 'pwd']), "Basic commands not found"
    print("✓ Get installed programs test passed")

def test_shell_singleton():
    """测试 shell 单例模式"""
    shell1 = get_shell()
    shell2 = get_shell()
    assert shell1 is shell2, "Shell singleton failed"
    
    # 测试设置新的 shell
    new_shell = ShellSession()
    set_shell(new_shell)
    assert get_shell() is new_shell, "Shell set failed"
    new_shell.close()
    print("✓ Shell singleton test passed")

def run_all_tests():
    """运行所有测试"""
    print("Running shell.py tests...")
    test_shell_basic_commands()
    test_shell_cd_command()
    test_split_commands()
    test_shorten_stdout()
    test_get_installed_programs()
    test_shell_singleton()
    print("\nAll tests passed! ✨")

if __name__ == "__main__":
    run_all_tests()
