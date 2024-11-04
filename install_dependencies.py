import subprocess
import sys
from typing import List, Tuple
import pkg_resources

def get_installed_packages() -> set:
    """获取已安装的包及其版本"""
    return {pkg.key for pkg in pkg_resources.working_set}

def read_requirements(files: List[str] = ['requirements.txt']) -> List[Tuple[str, str]]:
    """从requirements文件中读取依赖"""
    requirements = []
    for file in files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 处理带版本号的包
                        if '==' in line:
                            pkg, version = line.split('==')
                            requirements.append((pkg, version))
                        else:
                            requirements.append((line, ''))
        except FileNotFoundError:
            print(f"警告: {file} 文件未找到")
    return requirements

def check_and_install_package(package: str, version: str = '') -> None:
    """检查并安装包"""
    try:
        if version:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}=={version}'])
        else:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✓ 成功安装 {package}" + (f" (版本 {version})" if version else ""))
    except subprocess.CalledProcessError:
        print(f"✗ 安装 {package} 失败")

def auto_install_dependencies():
    """自动安装缺失的依赖"""
    # 基本必需的包
    essential_packages = [
        ('rich', ''),
        ('ipython', ''),
        ('pandas', ''),
        ('numpy', ''),
    ]
    
    installed_packages = get_installed_packages()
    
    # 尝试读取requirements.txt
    requirements = read_requirements()
    if requirements:
        packages_to_install = requirements
    else:
        packages_to_install = essential_packages
    
    print("检查依赖...")
    for package, version in packages_to_install:
        if package.lower() not in installed_packages:
            print(f"正在安装 {package}...")
            check_and_install_package(package, version)
        else:
            print(f"✓ {package} 已安装")

if __name__ == "__main__":
    print("开始检查并安装依赖...")
    auto_install_dependencies()
    print("依赖检查完成！") 