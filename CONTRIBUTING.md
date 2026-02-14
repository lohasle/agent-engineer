# 贡献指南

感谢您考虑为 agent-engineer 做出贡献！

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)

## 行为准则

本项目采用贡献者公约作为行为准则。参与此项目即表示您同意遵守其条款。

## 如何贡献

### 报告 Bug

如果您发现了 bug，请通过 [GitHub Issues](https://github.com/lohasle/agent-engineer/issues) 提交报告。提交时请包含：

1. 清晰的标题和描述
2. 复现步骤
3. 预期行为和实际行为
4. 您的环境信息（Python 版本、操作系统等）
5. 如有可能，提供最小化的代码示例

### 建议新功能

我们欢迎新功能建议！请创建一个 Issue 并详细描述：

1. 您希望实现的功能
2. 为什么这个功能对项目有用
3. 可能的实现方案

### 提交代码

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 进行代码更改
4. 确保通过所有测试
5. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
6. 推送到分支 (`git push origin feature/AmazingFeature`)
7. 创建 Pull Request

## 开发环境设置

### 前置要求

- Python 3.10+
- pip 或 poetry
- Git

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/lohasle/agent-engineer.git
cd agent-engineer

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e ".[dev]"

# 安装 pre-commit hooks
pre-commit install
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=. --cov-report=html

# 运行特定测试文件
pytest tests/test_agents.py -v
```

### 代码格式化

```bash
# 使用 black 格式化代码
black .

# 使用 ruff 检查代码
ruff check .

# 使用 mypy 进行类型检查
mypy graph_lib backend
```

## 代码规范

### Python 代码风格

- 遵循 [PEP 8](https://pep8.org/) 规范
- 使用 [Black](https://black.readthedocs.io/) 进行代码格式化
- 最大行长度：100 字符
- 使用类型注解

### 文档字符串

使用 Google 风格的文档字符串：

```python
def function_name(param1: str, param2: int) -> Dict[str, Any]:
    """函数的简短描述。

    更详细的描述（可选）。

    Args:
        param1: 第一个参数的描述。
        param2: 第二个参数的描述。

    Returns:
        返回值的描述。

    Raises:
        ValueError: 什么情况下会抛出此异常。
    """
    pass
```

### 命名约定

- 类名：`PascalCase`
- 函数/方法名：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有方法：`_leading_underscore`

## 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `style:` 代码格式（不影响代码运行的变动）
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建过程或辅助工具的变动

示例：
```
feat: add new evolutionary paradigm implementation
fix: resolve memory leak in agent execution
docs: update API documentation
```

## Pull Request 流程

1. 确保您的分支与 main 分支同步
2. 确保所有测试通过
3. 更新相关文档
4. 填写 PR 模板中的所有必要信息
5. 等待代码审查
6. 根据反馈进行必要的修改

## 贡献者

感谢所有贡献者！

---

*最后更新: 2026-02-15*
