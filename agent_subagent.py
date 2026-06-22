import os
import json
import subprocess
import sys
import glob as glob_module
from datetime import datetime
from openai import OpenAI

def load_settings():
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "settings.json")
    try:
        with open(settings_path, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"Error: Settings file not found at {settings_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in settings file: {e}")
        sys.exit(1)

settings = load_settings()

client = OpenAI(
    api_key=settings["api_key"],
    base_url=settings["base_url"],
)

MEMORY_FILE = "agent_memory.md"

# ==================== Basic Tool Calls ====================
def read(path, offset=None, limit=None):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        start = offset if offset else 0
        end = start + limit if limit else len(lines)
        return ''.join(f"{i+1:4d} {line}" for i, line in enumerate(lines[start:end], start))
    except Exception as e:
        return f"Error: {str(e)}"

def write(path, content):
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def edit(path, old_string, new_string):
    try:
        with open(path, 'r') as f:
            content = f.read()
        if content.count(old_string) != 1:
            return f"Error: old_string must appear exactly once (found {content.count(old_string)})"
        with open(path, 'w') as f:
            f.write(content.replace(old_string, new_string))
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def glob(pattern):
    try:
        files = sorted(glob_module.glob(pattern, recursive=True), key=lambda x: os.path.getmtime(x), reverse=True)
        return '\n'.join(files) if files else "No files found"
    except Exception as e:
        return f"Error: {str(e)}"

def grep(pattern, path="."):
    try:
        result = subprocess.run(f"grep -rn '{pattern}' {path}", shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else "No matches found"
    except Exception as e:
        return f"Error: {str(e)}"

def bash(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== SubAgent Implement(Main) ====================
def subagent(role, task):
    # Start an independent Agent loop, with its own dedicated role and separate context.
    print(f"\n{'='*50}")
    print(f"[SubAgent:{role}] Task: {task}")
    print(f"{'='*50}")

    sub_messages = [
        {"role": "system", "content": f"You are a {role}. Be concise and focused. Only do what is asked."},
        {"role": "user", "content": task}
    ]

    # SubAgent cannot spawn further subagents (to prevent infinite recursion), and only use basic tools.
    sub_tools = [t for t in tools if t["function"]["name"] != "subagent"]

    for _ in range(10):
        response = client.chat.completions.create(model=settings["model"],
                                                  messages=sub_messages,
                                                  tools=sub_tools)
        message = response.choices[0].message
        sub_messages.append(message)

        if not message.tool_calls:
            print(f"[SubAgent:{role}] finish\n")
            return message.content

        for tc in message.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [SubAgent:{role}] {fn}({json.dumps(args, ensure_ascii=False)[:80]})")
            result = available_functions[fn](**args)
            sub_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "SubAgent: max iterations reached"

# ================ Basic Tool Calls ================
available_functions = {
    "read": read, "write": write, "edit": edit,
    "glob": glob, "grep": grep, "bash": bash,
    "subagent": subagent,
}

tools = [
    {"type": "function",
     "function": {"name": "read",
                  "description": "Read file with line numbers",
                  "parameters": {"type": "object",
                                 "properties": {"path": {"type": "string"},
                                                "offset": {"type": "integer"},
                                                "limit": {"type": "integer"}},
                                 "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to file (creates dirs automatically)", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace a unique string in file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files by pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search files for pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "subagent", "description": "Delegate a task to a specialized sub-agent with its own role and independent context. Use this when a task requires specific expertise (e.g. 'frontend developer', 'DBA', 'test engineer').", "parameters": {"type": "object", "properties": {"role": {"type": "string", "description": "The sub-agent's specialty, e.g. 'Python backend developer'"}, "task": {"type": "string", "description": "The specific task to delegate"}}, "required": ["role", "task"]}}},
]

# ==================== Memory ====================
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return ""
    try:
        with open(MEMORY_FILE, 'r') as f:
            lines = f.read().split('\n')
        return '\n'.join(lines[-50:]) if len(lines) > 50 else '\n'.join(lines)
    except:
        return ""

def save_memory(task, result):
    try:
        with open(MEMORY_FILE, 'a') as f:
            f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Task:** {task}\n**Result:** {result}\n")
    except:
        pass

# ==================== Agent Core Loop ====================
def run_agent(messages, max_iterations=10):
    for _ in range(max_iterations):
        response = client.chat.completions.create(model=settings["model"],
                                                  messages=messages,
                                                  tools=tools)
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tc in message.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"[Tool] {fn}({json.dumps(args, ensure_ascii=False)[:100]})")
            result = available_functions.get(fn, lambda **_: f"Tool {fn} not found")(**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Max iterations reached"

# ==================== Main Entry ====================
def run(task):
    memory = load_memory()
    system = "You are an orchestrator agent. You can do tasks yourself or delegate to specialized sub-agents using the 'subagent' tool. Use subagent when a task benefits from focused expertise. Be concise."
    if memory:
        system += f"\n\n# Previous Context\n{memory}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task}
    ]

    result = run_agent(messages)
    print(f"\n{result}")
    save_memory(task, result)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 agent_subagent.py 'task'")
        print("\nExample:")
        print("  python3 agent_subagent.py '创建一个 TODO 应用，包含 Python 后端和 HTML 前端'")
        sys.exit(1)

    run(" ".join(sys.argv[1:]))
