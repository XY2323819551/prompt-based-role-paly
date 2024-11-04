#!/bin/bash
# 先运行依赖安装脚本
python install_dependencies.py

# 然后运行主程序
# python -m tool_pool.local_python_tool
python -m tool_pool.local_shell_tool
