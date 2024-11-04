from langchain_community.tools.bearly.tool import BearlyInterpreterTool
from tool_pool import BaseTool

class BearlyTool(BaseTool):
    def __init__(self, api_key: str):
        self.tool = BearlyInterpreterTool(api_key=api_key)
        super().__init__()

    def run(self, code: str):
        return self.tool.as_tool().run(code)
    
    def add_file(self, source_path: str, target_path: str, description: str):
        """添加文件到解释器环境"""
        self.tool.add_file(source_path, target_path, description)
    
    def clear_files(self):
        """清除所有已添加的文件"""
        self.tool.clear_files()

if __name__ == "__main__":
    # 需要替换为你的 Bearly API key
    API_KEY = "your_bearly_api_key"
    
    # 测试代码
    code = """
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.show()
    """
    
    tool = BearlyTool(API_KEY)
    print(tool.run(code)) 


    