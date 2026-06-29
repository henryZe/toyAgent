"""
Tests for agent_compact.py — covers compact_messages() and tool functions.

compact_messages() depends on client.chat.completions.create() for summarization,
so we mock the OpenAI client to avoid real API calls.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so agent_compact can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# agent_compact.py calls load_settings() at import time, which reads settings.json
# from the same directory as the module. We need to ensure that file exists.
_module_dir = os.path.dirname(os.path.abspath(
    os.path.join(os.path.join(os.path.dirname(__file__), ".."), "agent_compact.py")
))
_settings_path = os.path.join(_module_dir, "settings.json")

_settings_backup = None
if os.path.exists(_settings_path):
    _settings_backup = tempfile.mktemp(suffix=".json")
    import shutil
    shutil.copy2(_settings_path, _settings_backup)

if not os.path.exists(_settings_path):
    with open(_settings_path, "w") as f:
        json.dump({"api_key": "test-key", "base_url": "http://localhost:8080/", "model": "test-model"}, f)

import agent_compact

# Restore original settings
if _settings_backup:
    shutil.copy2(_settings_backup, _settings_path)
    os.remove(_settings_backup)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def mock_client():
    """
    Mock client.chat.completions.create to return a predictable summary.
    Returns the mock so tests can inspect call args.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mock summary of conversation"

    mock_create = MagicMock(return_value=mock_response)
    with patch.object(agent_compact.client.chat.completions, "create", mock_create):
        yield mock_create


def make_messages(count, system_content="You are a helpful assistant."):
    """Helper: build a list of `count` dict messages with alternating roles."""
    msgs = [{"role": "system", "content": system_content}]
    for i in range(count - 1):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}"})
    return msgs


# ==========================================================
# compact_messages() tests — below threshold
# ==========================================================

class TestCompactMessagesBelowThreshold:
    """When messages count <= COMPACT_THRESHOLD, no compression should occur."""

    def test_below_threshold_returns_original(self):
        msgs = make_messages(agent_compact.COMPACT_THRESHOLD)
        result = agent_compact.compact_messages(msgs)
        assert result == msgs

    def test_single_system_message_unchanged(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = agent_compact.compact_messages(msgs)
        assert result == msgs

    def test_five_messages_unchanged(self):
        msgs = make_messages(5)
        result = agent_compact.compact_messages(msgs)
        assert len(result) == 5


# ==========================================================
# compact_messages() tests — above threshold
# ==========================================================

class TestCompactMessagesAboveThreshold:
    """When messages count > COMPACT_THRESHOLD, compression should occur."""

    def test_compact_reduces_message_count(self, mock_client):
        msgs = make_messages(30)  # well above threshold
        result = agent_compact.compact_messages(msgs)
        # Expected: system + summary_user + summary_assistant + KEEP_RECENT
        assert len(result) == 1 + 2 + agent_compact.KEEP_RECENT

    def test_compact_preserves_system_message(self, mock_client):
        msgs = make_messages(30, system_content="Custom system prompt")
        result = agent_compact.compact_messages(msgs)
        assert result[0] == {"role": "system", "content": "Custom system prompt"}

    def test_compact_inserts_summary_pair(self, mock_client):
        msgs = make_messages(30)
        result = agent_compact.compact_messages(msgs)
        # Position 1 should be the summary user message
        assert result[1]["role"] == "user"
        assert "[Previous conversation summary]" in result[1]["content"]
        # Position 2 should be the handshake assistant message
        assert result[2]["role"] == "assistant"
        assert "Understood" in result[2]["content"]

    def test_compact_preserves_recent_messages(self, mock_client):
        msgs = make_messages(30)
        result = agent_compact.compact_messages(msgs)
        # The last KEEP_RECENT messages should be unchanged
        recent_original = msgs[-agent_compact.KEEP_RECENT:]
        recent_result = result[-agent_compact.KEEP_RECENT:]
        assert recent_result == recent_original

    def test_compact_calls_llm_for_summary(self, mock_client):
        msgs = make_messages(30)
        agent_compact.compact_messages(msgs)
        mock_client.assert_called_once()
        # Verify it was called with summarization system prompt
        call_args = mock_client.call_args
        call_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert "Summarize" in call_messages[0]["content"]

    def test_compact_summary_contains_llm_response(self, mock_client):
        msgs = make_messages(30)
        result = agent_compact.compact_messages(msgs)
        assert "Mock summary of conversation" in result[1]["content"]

    def test_compact_exactly_at_threshold_plus_one(self, mock_client):
        # Just one above threshold
        msgs = make_messages(agent_compact.COMPACT_THRESHOLD + 1)
        result = agent_compact.compact_messages(msgs)
        assert len(result) == 1 + 2 + agent_compact.KEEP_RECENT

    def test_compact_with_tool_messages(self, mock_client):
        """Messages with tool role should be handled correctly."""
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(agent_compact.COMPACT_THRESHOLD):
            if i % 3 == 0:
                msgs.append({"role": "user", "content": f"Task {i}"})
            elif i % 3 == 1:
                msgs.append({"role": "assistant", "content": f"Response {i}"})
            else:
                msgs.append({"role": "tool", "tool_call_id": f"tc_{i}", "content": f"Tool result {i}"})
        # Add enough to exceed threshold
        msgs.append({"role": "user", "content": "Extra task"})
        msgs.append({"role": "assistant", "content": "Extra response"})

        result = agent_compact.compact_messages(msgs)
        assert len(result) == 1 + 2 + agent_compact.KEEP_RECENT
        assert result[0]["role"] == "system"


# ==========================================================
# Tool function tests
# ==========================================================

class TestToolFunctions:
    """Test the 3 basic tool functions independently."""

    def test_execute_bash_simple_command(self):
        result = agent_compact.execute_bash("echo hello")
        assert "hello" in result

    def test_execute_bash_error_command(self):
        result = agent_compact.execute_bash("ls /nonexistent_dir_xyz")
        assert result  # Should contain stderr output

    def test_execute_bash_timeout(self):
        result = agent_compact.execute_bash("sleep 31")
        assert "Error" in result

    def test_read_file_success(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        result = agent_compact.read_file(str(test_file))
        assert result == "Hello World"

    def test_read_file_nonexistent(self):
        result = agent_compact.read_file("/nonexistent_file_xyz.txt")
        assert "Error" in result

    def test_write_file_success(self, tmp_path):
        test_file = tmp_path / "output.txt"
        result = agent_compact.write_file(str(test_file), "Test content")
        assert "Successfully" in result
        assert test_file.read_text() == "Test content"

    def test_write_file_overwrite(self, tmp_path):
        test_file = tmp_path / "overwrite.txt"
        test_file.write_text("Old content")
        agent_compact.write_file(str(test_file), "New content")
        assert test_file.read_text() == "New content"


# ==========================================================
# Edge cases
# ==========================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_messages_with_none_content(self, mock_client):
        """Some messages may have None content (e.g., assistant with tool_calls)."""
        msgs = [{"role": "system", "content": "System"}]
        msgs.append({"role": "user", "content": "Task"})
        msgs.append({"role": "assistant", "content": None})  # tool_calls only, no text
        msgs.append({"role": "tool", "tool_call_id": "tc_1", "content": "Result"})
        # Fill to exceed threshold
        for i in range(agent_compact.COMPACT_THRESHOLD):
            msgs.append({"role": "user", "content": f"Msg {i}"})
            msgs.append({"role": "assistant", "content": f"Ans {i}"})

        result = agent_compact.compact_messages(msgs)
        assert result[0]["role"] == "system"

    def test_messages_with_empty_content(self, mock_client):
        """Messages with empty string content."""
        msgs = [{"role": "system", "content": "System"}]
        msgs.append({"role": "user", "content": ""})  # empty content
        for i in range(agent_compact.COMPACT_THRESHOLD + 1):
            msgs.append({"role": "assistant", "content": f"Ans {i}"})
            msgs.append({"role": "user", "content": f"Msg {i}"})

        result = agent_compact.compact_messages(msgs)
        assert result[0]["role"] == "system"

    def test_compact_threshold_boundary_exact(self):
        """Exactly COMPACT_THRESHOLD messages — should NOT compress."""
        msgs = make_messages(agent_compact.COMPACT_THRESHOLD)
        result = agent_compact.compact_messages(msgs)
        assert result == msgs  # no change

    def test_compact_threshold_boundary_plus_one(self, mock_client):
        """COMPACT_THRESHOLD + 1 messages — should compress."""
        msgs = make_messages(agent_compact.COMPACT_THRESHOLD + 1)
        result = agent_compact.compact_messages(msgs)
        assert result != msgs
        assert len(result) < len(msgs)
