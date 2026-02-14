"""Tests for Agent Engineer Architecture."""

import pytest
from graph_lib.agents import (
    AgentConfig,
    AgentState,
    Message,
    MessageType,
)


class TestAgentConfig:
    def test_default_config(self):
        config = AgentConfig()
        assert config.name == ""
        assert config.model == "gpt-4"
        assert config.max_iterations == 10
        assert config.tools == []

    def test_custom_config(self):
        config = AgentConfig(
            name="Test Agent",
            model="gpt-3.5-turbo",
            max_iterations=5,
            tools=["search", "calculator"],
        )
        assert config.name == "Test Agent"
        assert config.model == "gpt-3.5-turbo"
        assert config.max_iterations == 5
        assert "search" in config.tools

    def test_to_dict(self):
        config = AgentConfig(name="Test", model="gpt-4")
        result = config.to_dict()
        assert result["name"] == "Test"
        assert result["model"] == "gpt-4"


class TestMessage:
    def test_message_creation(self):
        message = Message(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.TASK,
            content={"task": "test"},
        )
        assert message.sender_id == "agent1"
        assert message.receiver_id == "agent2"
        assert message.message_type == MessageType.TASK

    def test_message_to_dict(self):
        message = Message(
            sender_id="agent1",
            content="test content",
        )
        result = message.to_dict()
        assert result["sender_id"] == "agent1"
        assert result["content"] == "test content"

    def test_message_from_dict(self):
        data = {
            "sender_id": "agent1",
            "receiver_id": "agent2",
            "message_type": "task",
            "content": "test",
        }
        message = Message.from_dict(data)
        assert message.sender_id == "agent1"
        assert message.message_type == MessageType.TASK


class TestAgentState:
    def test_state_values(self):
        assert AgentState.IDLE == "idle"
        assert AgentState.THINKING == "thinking"
        assert AgentState.ACTING == "acting"
        assert AgentState.ERROR == "error"
        assert AgentState.COMPLETED == "completed"
