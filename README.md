## 项目概述

本项目包含 7 个渐进式、独立的 AI Agent 实现，每个演示一种不同的 Agent 架构范式。所有 Agent 使用通过 `settings.json` 配置的 OpenAI 兼容聊天补全 API。

## 运行 Agent

所有 Agent 脚本需要 `settings.json` 中配置 `api_key`、`base_url` 和 `model`。每个脚本独立运行，无共享库依赖。

```
python agent.py "task"                              # 基础 ReAct，最大 5 次迭代
python agent_plus.py [--plan] "task"                # +记忆持久化、任务规划，最大 500 次迭代
python agent_claude.py [--plan] "task"              # Claude-Code 风格：7 个工具、plan-as-tool、rules/skills/MCP
python agent_compact.py "task"                      # +上下文窗口压缩（COMPACT_THRESHOLD=20, KEEP_RECENT=6）
python agent_safe.py [--auto] "task"                # +3 层安全护栏：命令黑名单、用户确认、输出截断
python agent_subagent.py "task"                     # 编排器通过 subagent 工具委派给专业子 Agent
python agent_team.py "task"                         # 团队协作：plan_team → hire → broadcast → review → disband
```

## 测试

`pytest.ini` 配置 `testpaths = tests python_api/tests` 和 `pythonpath = . python_api`。Agent 测试使用 `tests/settings.json`（模拟凭据）并 mock OpenAI 客户端。测试集中在 `tests/test_agent_compact.py`，覆盖 `compact_messages()` 压缩逻辑和工具函数（execute_bash、read_file、write_file）。

```
pytest                              # 运行全部测试
pytest tests/                       # 仅 Agent 测试
pytest -k "test_name"               # 按名称匹配运行单个测试（如 -k "compact_reduces"）
pytest -k "compact"                 # 按名称匹配运行所有压缩相关测试
pytest -k "execute_bash"            # 按名称匹配运行所有 bash 工具测试
pytest tests/test_agent_compact.py  # 指定文件运行
pytest -v                           # 详细输出，显示每个测试名称
pytest -s                           # 显示 print/stdout 输出（默认捕获）
pytest --tb=short                   # 简短回溯格式
pytest --tb=long                    # 完整回溯格式（默认）
pytest -x                           # 遇到第一个失败即停止
pytest --lf                         # 仅重新运行上次失败的测试
pytest -vv                          # 更详细输出（显示测试完整路径）
```

## Agent 架构

每个 Agent 文件是独立的、自包含的实现，在前一版本概念基础上构建。关键架构模式：

- **工具接口**：所有工具为包含 `type`、`function`（名称、描述、参数 JSON Schema）及本地 `_fn` 实现的字典。LLM 通过 OpenAI function-calling 格式选择工具；Agent 循环分发至对应的 `_fn`。
- **Agent 循环模式**：`while iterations < max: 发送消息 → 获取响应 → 若 tool_call: 执行工具 → 追加结果 → 继续; 否则: 返回最终文本`
- **记忆持久化**：Agent 向 `agent_memory.md` 追加内容并在启动时加载最后 50 行（`agent.py` 和 `agent_team.py` 除外）。
- **工具演进**：v1/v2/v4/v5/v6 使用 3 个工具（execute_bash, read_file, write_file）。v3/v7 扩展至 7 个（read, write, edit, glob, grep, bash, plan/subagent），镜像 Claude Code 的接口。`edit` 工具使用唯一字符串匹配语义。
- **子 Agent 委派**（`agent_subagent.py`）：编排器使用 `subagent` 工具在独立上下文中生成专注型 Agent。子 Agent 不能递归生成更多子 Agent。
- **团队协作**（`agent_team.py`）：Team 类管理带有收件箱的持久 Agent 对象。通过 `send()`（直接）和 `broadcast()`（全员）通信。LLM 决定团队组成。
- **上下文压缩**（`agent_compact.py`）：当消息超过阈值时，旧消息由 LLM 概括为一对消息，保留近期消息不变。
- **安全护栏**（`agent_safe.py`）：危险命令黑名单、用户 Y/N/Q 确认、输出截断至 5000 字符。
