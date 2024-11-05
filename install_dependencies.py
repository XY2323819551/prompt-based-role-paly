import subprocess
import sys
from typing import List, Tuple, Set, Dict
import pkg_resources
import importlib.util
import ast
import os

def get_installed_packages() -> Set[str]:
    """获取已安装的包及其版本"""
    return {pkg.key for pkg in pkg_resources.working_set}

class ImportVisitor(ast.NodeVisitor):
    """用于分析Python文件中的导入语句"""
    def __init__(self):
        self.imports = set()
        self.local_imports = set()

    def visit_Import(self, node):
        for name in node.names:
            base_module = name.name.split('.')[0]
            if self._is_local_module(base_module):
                self.local_imports.add(base_module)
            else:
                self.imports.add(base_module)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if self._is_local_module(base_module):
                self.local_imports.add(base_module)
            else:
                self.imports.add(base_module)

    def _is_local_module(self, module_name: str) -> bool:
        """
        判断是否为本地模块
        """
        # 检查当前目录及其子目录是否存在该模块
        for root, dirs, files in os.walk('.'):
            if module_name in dirs:  # 检查是否为本地包
                return True
            # 检查是否为本地模块文件
            if f"{module_name}.py" in files:
                return True
        return False

def get_file_dependencies(file_path: str) -> Tuple[Set[str], Set[str]]:
    """
    分析Python文件的导入依赖
    
    Args:
        file_path: Python文件的路径
    
    Returns:
        Tuple[Set[str], Set[str]]: (第三方依赖包集合, 本地模块集合)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read())
        visitor = ImportVisitor()
        visitor.visit(tree)
        
    # 过滤掉Python标准库
    stdlib_modules = set(sys.stdlib_module_names)
    third_party_imports = {imp for imp in visitor.imports if imp not in stdlib_modules}
    
    return third_party_imports, visitor.local_imports

# 常用包的映射关系
PACKAGE_MAPPING = {
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'bs4': 'beautifulsoup4',
    'yaml': 'pyyaml',
}

def resolve_dependencies(imports: Set[str]) -> Set[str]:
    """解析导入名称到实际的包名"""
    resolved = set()
    for imp in imports:
        resolved.add(PACKAGE_MAPPING.get(imp, imp))
    return resolved

def install_package(package: str) -> bool:
    """安装单个包"""
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet', package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ 安装 {package} 失败")
        return False

def check_and_install_for_file(file_path: str) -> bool:
    """
    检查并安装Python文件所需的依赖
    
    Args:
        file_path: Python文件的路径
    
    Returns:
        bool: 是否所有依赖都安装成功
    """
    print(f"正在分析文件依赖: {file_path}")
    
    # 获取文件的依赖
    third_party_imports, local_imports = get_file_dependencies(file_path)
    
    if local_imports:
        print(f"检测到本地模块: {', '.join(local_imports)}")
    
    required_packages = resolve_dependencies(third_party_imports)
    
    # 检查已安装的包
    installed_packages = get_installed_packages()
    packages_to_install = {
        pkg for pkg in required_packages 
        if pkg.lower() not in installed_packages
    }
    
    if not packages_to_install:
        print("✓ 所有第三方依赖已安装")
        return True
    
    print(f"需要安装的第三方依赖: {', '.join(packages_to_install)}")
    
    # 安装缺失的依赖
    all_success = True
    for package in packages_to_install:
        if not install_package(package):
            all_success = False
    
    return all_success

def get_module_file_path(module_path: str) -> str:
    """
    将模块路径转换为文件路径
    
    Args:
        module_path: 模块路径 (例如: 'tool_pool.arxiv_pdf')
    
    Returns:
        str: 文件路径
    """
    parts = module_path.split('.')
    file_path = os.path.join(*parts) + '.py'
    return file_path

def main():
    """主函数"""
    if len(sys.argv) > 1:
        module_path = sys.argv[1]
        file_path = get_module_file_path(module_path)
        
        if not os.path.exists(file_path):
            print(f"错误: 找不到文件 {file_path}")
            sys.exit(1)
            
        success = check_and_install_for_file(file_path)
        if not success:
            sys.exit(1)
    else:
        print("请指定要检查的Python模块路径")
        sys.exit(1)

if __name__ == "__main__":
    main() 