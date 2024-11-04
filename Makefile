# 定义清理目标
.PHONY: clean

# 清理命令
clean:
	@echo "Cleaning __pycache__ directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleaning .pyc files..."
	@find . -type f -name "*.pyc" -delete
	@echo "Cleaning complete."

# 统计Python文件数量和代码行数
.PHONY: count
count:
	@echo "Counting Python files and lines of code..."
	@echo "Python files count:" && find . -type f -name "*.py" | wc -l
	@echo "Total lines in Python files:" && find . -type f -name "*.py" -exec cat {} \; | wc -l
