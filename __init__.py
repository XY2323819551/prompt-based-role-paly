from pathlib import Path
import sys

# 获取项目根目录
project_root = Path(__file__).resolve().parent

# 将项目根目录添加到Python路径
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))




