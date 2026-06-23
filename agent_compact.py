import os
import json
import subprocess
import sys
from openai import OpenAI

def load_settings():
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "settings.json")
    with open(settings_path, "r") as f:
        return json.load(f)

settings = load_settings()

client = OpenAI(
    api_key=settings["api_key"],
    base_url=settings["base_url"],
)

# ==================== 工具（和 agent.py 一样） ====================

tools = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Execute a bash command on the system", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read contents of a file", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["path", "content"]}}},
]

def execute_bash(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

def write_file(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"

available_functions = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file
}

# ==================== 上下文压缩（核心新增） ====================
#
# 就这一个函数，约 30 行，实现了完整的压缩逻辑。
#
# 原理：
#   messages = [system, user, assistant, tool, assistant, tool, ...]
#                 ↓ 压缩后
#   messages = [system, 摘要(包含之前所有要点), 最近 N 条消息]

COMPACT_THRESHOLD = 20  # messages 超过这个数量就触发压缩
KEEP_RECENT = 6         # 压缩时保留最近几条消息（不压缩）

def compact_messages(messages):
    """
    当 messages 过长时，把旧消息压缩成一段摘要。

    压缩前: [system, msg1, msg2, ..., msg15, msg16, msg17, msg18, msg19, msg20]
    压缩后: [system, summary_of(msg1~msg14), msg15, msg16, msg17, msg18, msg19, msg20]
    """
    if len(messages) <= COMPACT_THRESHOLD:
        return messages  # 没超阈值，不压缩

    print(f"\n[Compact] messages 数量 ({len(messages)}) 超过阈值 ({COMPACT_THRESHOLD})，开始压缩...")

    system_msg = messages[0]            # system prompt 永远保留
    old_messages = messages[1:-KEEP_RECENT]  # 需要被压缩的旧消息
    recent_messages = messages[-KEEP_RECENT:]  # 最近的消息保留原样

    # 把旧消息拼成文本，交给 LLM 做摘要
    old_text = ""
    for msg in old_messages:
        role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content:
            old_text += f"[{role}]: {content}\n"

    # 调用 LLM 生成摘要
    summary_response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {"role": "system", "content": "Summarize the following conversation history into a concise summary. Keep all important facts, file paths, command results, and decisions. Be concise but don't lose critical details."},
            {"role": "user", "content": old_text}
        ]
    )
    summary = summary_response.choices[0].message.content

    print(f"[Compact] {len(old_messages)} 条旧消息 → 1 条摘要 (保留最近 {len(recent_messages)} 条)")
    print(f"[Compact] 压缩后 messages: {1 + 1 + len(recent_messages)} 条\n")

    # 重新组装：system + 摘要 + 最近消息
    return [
        system_msg,
        {"role": "user", "content": f"[Previous conversation summary]: {summary}"},
        {"role": "assistant", "content": "Understood. I have the context from our previous conversation. Let me continue."},
        *recent_messages
    ]

# ==================== Agent 核心循环（在 agent.py 基础上加了压缩） ====================

def run_agent(user_message, max_iterations=30):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can interact with the system. Be concise."},
        {"role": "user", "content": user_message}
    ]

    for i in range(max_iterations):
        # ===== 新增：每轮循环前检查是否需要压缩 =====
        messages = compact_messages(messages)

        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {function_name}({json.dumps(function_args, ensure_ascii=False)[:80]})")
            function_response = available_functions[function_name](**function_args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})

    return "Max iterations reached"

# ==================== 主入口 ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 agent_compact.py 'task'")
        print("\nExample:")
        print("  python3 agent_compact.py '找到所有 Python 文件，统计行数，按行数排序，写入 report.txt'")
        sys.exit(1)
    result = run_agent(" ".join(sys.argv[1:]))
    print(f"\n{result}")
