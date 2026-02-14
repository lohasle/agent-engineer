"""
Base Agent Classes for Agent Engineer Architecture.

This module provides the foundational classes for all agent types
in the LangGraph-based agent collaboration system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
import json


class AgentState(str, Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class MessageType(str, Enum):
    """Types of messages between agents."""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ERROR = "error"
    SYNC = "sync"


@dataclass
class Message:
    """Message structure for agent communication."""
    id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.TASK
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            id=data.get("id", str(uuid4())),
            sender_id=data.get("sender_id", ""),
            receiver_id=data.get("receiver_id", ""),
            message_type=MessageType(data.get("message_type", "task")),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", __import__('time').time())
        )


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    name: str = ""
    description: str = ""
    model: str = "gpt-4"
    max_iterations: int = 10
    timeout: int = 300
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    memory_size: int = 100
    checkpoint_interval: int = 5
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "memory_size": self.memory_size,
            "checkpoint_interval": self.checkpoint_interval,
            "debug_mode": self.debug_mode
        }


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = str(uuid4())
        self.state = AgentState.IDLE
        self.memory: List[Dict[str, Any]] = []
        self.tools: Dict[str, Callable] = {}
        self.message_queue: List[Message] = []
        self.iteration_count = 0
        self.checkpoint_count = 0
        
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process information and plan next actions."""
        pass
    
    @abstractmethod
    async def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planned actions."""
        pass
    
    @abstractmethod
    async def observe(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process action results and update understanding."""
        pass
    
    async def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full think-act-observe cycle."""
        self.state = AgentState.THINKING
        context = initial_input
        
        for _ in range(self.config.max_iterations):
            self.iteration_count += 1
            
            # Think phase
            plan = await self.think(context)
            
            # Check for completion
            if plan.get("action") == "complete":
                self.state = AgentState.COMPLETED
                return {"status": "completed", "result": plan.get("result")}
            
            # Act phase
            self.state = AgentState.ACTING
            result = await self.act(plan)
            
            # Observe phase
            self.state = AgentState.OBSERVING
            observation = await self.observe(result)
            context.update(observation)
            
            # Checkpoint
            if self.iteration_count % self.config.checkpoint_interval == 0:
                await self._save_checkpoint()
        
        self.state = AgentState.ERROR
        return {"status": "timeout", "message": "Max iterations reached"}
    
    async def send_message(self, receiver_id: str, content: Any, 
                          message_type: MessageType = MessageType.TASK) -> None:
        """Send a message to another agent."""
        message = Message(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        # In a real implementation, this would use a message broker
        self.message_queue.append(message)
    
    async def receive_message(self, message: Message) -> None:
        """Receive a message from another agent."""
        self.memory.append({
            "type": "message",
            "from": message.sender_id,
            "content": message.content,
            "timestamp": message.timestamp
        })
    
    def add_tool(self, name: str, func: Callable) -> None:
        """Register a tool for this agent."""
        self.tools[name] = func
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return await self.tools[tool_name](**kwargs)
    
    async def _save_checkpoint(self) -> None:
        """Save agent state checkpoint."""
        self.checkpoint_count += 1
        checkpoint = {
            "iteration": self.iteration_count,
            "checkpoint": self.checkpoint_count,
            "state": self.state.value,
            "memory": self.memory[-self.config.memory_size:]
        }
        # In a real implementation, this would persist to storage
        if self.config.debug_mode:
            print(f"Checkpoint saved: {json.dumps(checkpoint, indent=2)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "id": self.id,
            "name": self.config.name,
            "state": self.state.value,
            "iterations": self.iteration_count,
            "checkpoints": self.checkpoint_count,
            "memory_size": len(self.memory)
        }


class ManagerAgent(BaseAgent):
    """Manager agent for hierarchical agent structures."""
    
    def __init__(self, config: AgentConfig, workers: List[BaseAgent] = None):
        super().__init__(config)
        self.workers = workers or []
        self.task_queue: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task and delegate to workers."""
        task = context.get("task", "")
        subtasks = await self._decompose_task(task)
        
        for subtask in subtasks:
            worker = self._select_worker(subtask)
            if worker:
                self.task_queue.append({
                    "task": subtask,
                    "worker_id": worker.id,
                    "status": "pending"
                })
        
        return {"action": "delegate", "tasks": self.task_queue}
    
    async def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task delegation."""
        results = {}
        for task_info in self.task_queue:
            if task_info["status"] == "pending":
                worker = next((w for w in self.workers if w.id == task_info["worker_id"]), None)
                if worker:
                    result = await worker.run({"task": task_info["task"]})
                    results[task_info["task"]] = result
                    task_info["status"] = "completed"
        return {"action": "delegate", "results": results}
    
    async def observe(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process worker results."""
        self.results.update(action_result.get("results", {}))
        return {"results": self.results}
    
    async def _decompose_task(self, task: str) -> List[str]:
        """Decompose a task into subtasks."""
        # In a real implementation, this would use LLM for decomposition
        return [task]
    
    def _select_worker(self, subtask: str) -> Optional[BaseAgent]:
        """Select the best worker for a subtask."""
        # Simple round-robin for now
        if self.workers:
            return self.workers[len(self.task_queue) % len(self.workers)]
        return None


class PeerAgent(BaseAgent):
    """Peer agent for collaborative agent structures."""
    
    def __init__(self, config: AgentConfig, peers: List[BaseAgent] = None):
        super().__init__(config)
        self.peers = peers or []
        self.collaboration_strategy = "debate"
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with peers to solve problems."""
        task = context.get("task", "")
        if self.peers and self.collaboration_strategy == "debate":
            return await self._debate_strategy(task)
        return {"action": "independent", "task": task}
    
    async def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute based on collaboration strategy."""
        if plan.get("action") == "debate":
            return await self._execute_debate(plan)
        return await self._execute_independent(plan)
    
    async def observe(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process collaboration results."""
        return {"result": action_result}
    
    async def _debate_strategy(self, task: str) -> Dict[str, Any]:
        """Prepare for debate with peers."""
        for peer in self.peers:
            await self.send_message(
                peer.id,
                {"topic": task, "role": "debater"},
                MessageType.QUERY
            )
        return {"action": "debate", "topic": task}
    
    async def _execute_debate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute debate with peers."""
        results = []
        for peer in self.peers:
            response = await self._wait_for_response(peer.id)
            results.append(response)
        return {"action": "debate", "results": results}
    
    async def _execute_independent(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task independently."""
        # Implement independent task execution
        return {"action": "independent", "result": "completed"}
    
    async def _wait_for_response(self, peer_id: str, timeout: int = 30) -> Any:
        """Wait for a response from a peer."""
        # Simplified implementation
        return {"peer_id": peer_id, "contribution": "placeholder"}
