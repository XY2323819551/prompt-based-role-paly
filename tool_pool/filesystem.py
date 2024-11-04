from langchain_community.agent_toolkits import FileManagementToolkit
from tool_pool import BaseTool
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class FileSystemResult:
    """文件系统操作结果的数据类"""
    success: bool
    message: str
    content: Optional[str] = None
    file_list: Optional[List[str]] = None


class FileSystemTool(BaseTool):
    def __init__(self, root_dir: Optional[str] = None):
        """
        初始化文件系统工具。

        Args:
            root_dir (str, optional): 根目录路径。如果不提供，将使用当前工作目录。
        """
        self.toolkit = FileManagementToolkit(
            root_dir=root_dir,
            selected_tools=["read_file", "write_file", "list_directory", "file_delete"]
        )
        self.tools = {
            tool.__class__.__name__: tool 
            for tool in self.toolkit.get_tools()
        }
        super().__init__()

    def read_file(self, file_path: str) -> FileSystemResult:
        """
        读取文件内容。

        Args:
            file_path (str): 文件路径

        Returns:
            FileSystemResult: 操作结果
        """
        try:
            content = self.tools['ReadFileTool'].run({"file_path": file_path})
            return FileSystemResult(
                success=True,
                message="File read successfully",
                content=content
            )
        except Exception as e:
            return FileSystemResult(
                success=False,
                message=f"Error reading file: {str(e)}"
            )

    def write_file(self, file_path: str, content: str) -> FileSystemResult:
        """
        写入文件内容。

        Args:
            file_path (str): 文件路径
            content (str): 要写入的内容

        Returns:
            FileSystemResult: 操作结果
        """
        try:
            result = self.tools['WriteFileTool'].run({
                "file_path": file_path,
                "text": content
            })
            return FileSystemResult(
                success=True,
                message=result
            )
        except Exception as e:
            return FileSystemResult(
                success=False,
                message=f"Error writing file: {str(e)}"
            )

    def list_directory(self) -> FileSystemResult:
        """
        列出目录内容。

        Returns:
            FileSystemResult: 操作结果，包含文件列表
        """
        try:
            files = self.tools['ListDirectoryTool'].run({})
            return FileSystemResult(
                success=True,
                message="Directory listed successfully",
                file_list=files.split('\n') if isinstance(files, str) else files
            )
        except Exception as e:
            return FileSystemResult(
                success=False,
                message=f"Error listing directory: {str(e)}"
            )

    def delete_file(self, file_path: str) -> FileSystemResult:
        """
        删除文件。

        Args:
            file_path (str): 要删除的文件路径

        Returns:
            FileSystemResult: 操作结果
        """
        try:
            result = self.tools['DeleteFileTool'].run({"file_path": file_path})
            return FileSystemResult(
                success=True,
                message=result
            )
        except Exception as e:
            return FileSystemResult(
                success=False,
                message=f"Error deleting file: {str(e)}"
            )


if __name__ == "__main__":
    # 测试文件系统工具
    from tempfile import TemporaryDirectory
    
    # 创建临时目录进行测试
    with TemporaryDirectory() as temp_dir:
        fs_tool = FileSystemTool(root_dir=temp_dir)
        
        # 测试写入文件
        print("Testing write_file:")
        write_result = fs_tool.write_file("test.txt", "Hello, World!")
        print(write_result)
        
        # 测试列出目录
        print("\nTesting list_directory:")
        list_result = fs_tool.list_directory()
        print(list_result)
        
        # 测试读取文件
        print("\nTesting read_file:")
        read_result = fs_tool.read_file("test.txt")
        print(read_result)
        
        # 测试删除文件
        print("\nTesting delete_file:")
        delete_result = fs_tool.delete_file("test.txt")
        print(delete_result)
        
        # 确认文件已被删除
        print("\nVerifying deletion:")
        final_list = fs_tool.list_directory()
        print(final_list) 