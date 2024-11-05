#!/bin/bash

# 获取要运行的模块名
MODULE_PATH=$1

if [ -z "$MODULE_PATH" ]; then
    echo "请指定要运行的模块路径"
    exit 1
fi

# 先检查并安装依赖
python install_dependencies.py "$MODULE_PATH"

# 如果依赖安装成功，运行指定模块
if [ $? -eq 0 ]; then
    echo "依赖检查完成，开始运行模块..."
    python -m $MODULE_PATH
else
    echo "依赖安装失败，请检查错误信息"
    exit 1
fi
