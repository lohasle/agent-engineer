"""
Backend API for Agent Engineer Architecture.

This module provides a Flask/FastAPI backend for agent generation,
graph management, and real-time interaction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum


app = Flask(__name__)
CORS(app)


class AgentStatus(str, Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentExecution:
    """Agent execution context."""
    id: str
    config: Dict[str, Any]
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


# In-memory storage for demo (use database in production)
agent_executions: Dict[str, AgentExecution] = {}
executor = ThreadPoolExecutor(max_workers=4)


def get_paradigm_info() -> List[Dict[str, str]]:
    """Get information about available paradigms."""
    return [
        {
            "id": "react",
            "name": "React",
            "description": "Standard think-act-observe loop for reasoning",
            "complexity": "low",
            "use_cases": ["Single agent tasks", "Sequential reasoning", "Simple automation"]
        },
        {
            "id": "surprise",
            "name": "Surprise",
            "description": "Self-improving agent that learns from unexpected outcomes",
            "complexity": "medium",
            "use_cases": ["Adaptive learning", "Error recovery", "Strategy optimization"]
        },
        {
            "id": "peer",
            "name": "Peer",
            "description": "Multi-agent collaboration through debate and synthesis",
            "complexity": "medium",
            "use_cases": ["Collaborative problem solving", "Multi-perspective analysis", "Consensus building"]
        },
        {
            "id": "hierarchical",
            "name": "Hierarchical",
            "description": "Manager-worker structure for task distribution",
            "complexity": "high",
            "use_cases": ["Complex workflows", "Task delegation", "Scalable processing"]
        },
        {
            "id": "evolutionary",
            "name": "Evolutionary",
            "description": "Population-based evolution for continuous improvement",
            "complexity": "high",
            "use_cases": ["Optimization", "Strategy discovery", "Adaptive systems"]
        }
    ]


def get_tools_list() -> List[Dict[str, str]]:
    """Get available tools for agents."""
    return [
        {"id": "search", "name": "Web Search", "description": "Search the web for information"},
        {"id": "browser", "name": "Browser", "description": "Navigate and interact with web pages"},
        {"id": "calculator", "name": "Calculator", "description": "Perform mathematical calculations"},
        {"id": "file_read", "name": "File Read", "description": "Read files from the filesystem"},
        {"id": "file_write", "name": "File Write", "description": "Write files to the filesystem"},
        {"id": "code_execute", "name": "Code Execute", "description": "Execute code snippets"},
        {"id": "api_call", "name": "API Call", "description": "Make HTTP API requests"},
        {"id": "database", "name": "Database", "description": "Query and update databases"},
        {"id": "email", "name": "Email", "description": "Send and receive emails"},
        {"id": "calendar", "name": "Calendar", "description": "Manage calendar events"}
    ]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "agent-engineer-backend",
        "version": "1.0.0"
    })


@app.route('/api/paradigms', methods=['GET'])
def list_paradigms():
    """List available agent paradigms."""
    return jsonify({
        "paradigms": get_paradigm_info()
    })


@app.route('/api/tools', methods=['GET'])
def list_tools():
    """List available tools."""
    return jsonify({
        "tools": get_tools_list()
    })


@app.route('/api/generate', methods=['POST'])
def generate_agent():
    """Generate LangGraph code from configuration."""
    try:
        config = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description', 'paradigm']
        for field in required_fields:
            if field not in config:
                return jsonify({
                    "error": f"Missing required field: {field}"
                }), 400
        
        # Import generator (lazy import to avoid startup delays)
        from generator import generate_agent
        
        # Generate code
        files = generate_agent(config)
        
        return jsonify({
            "status": "success",
            "agent_id": str(uuid.uuid4()),
            "files": {
                "agent.py": files.get("agent", ""),
                "graph.py": files.get("graph", ""),
                "config.json": files.get("config", ""),
                "requirements.txt": files.get("requirements", "")
            }
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/generate/code', methods=['POST'])
def generate_code_only():
    """Generate code only without saving."""
    try:
        config = request.get_json()
        from generator import generate_agent
        
        files = generate_agent(config)
        
        return jsonify({
            "status": "success",
            "code": files.get("agent", "")
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/execute', methods=['POST'])
def execute_agent():
    """Execute an agent with input."""
    try:
        data = request.get_json()
        agent_id = str(uuid.uuid4())
        
        config = data.get("config", {})
        input_data = data.get("input", {})
        
        execution = AgentExecution(
            id=agent_id,
            config=config,
            status=AgentStatus.PENDING,
            created_at=__import__('time').time()
        )
        
        agent_executions[agent_id] = execution
        
        # Run in background
        executor.submit(_run_agent_background, agent_id, config, input_data)
        
        return jsonify({
            "execution_id": agent_id,
            "status": "started"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/execute/<execution_id>', methods=['GET'])
def get_execution_status(execution_id: str):
    """Get execution status and result."""
    execution = agent_executions.get(execution_id)
    
    if not execution:
        return jsonify({
            "error": "Execution not found"
        }), 404
    
    response = {
        "execution_id": execution.id,
        "status": execution.status.value,
        "config": execution.config,
        "created_at": execution.created_at,
        "completed_at": execution.completed_at
    }
    
    if execution.result:
        response["result"] = execution.result
    
    if execution.error:
        response["error"] = execution.error
    
    return jsonify(response)


@app.route('/api/execute/<execution_id>', methods=['DELETE'])
def cancel_execution(execution_id: str):
    """Cancel a running execution."""
    execution = agent_executions.get(execution_id)
    
    if not execution:
        return jsonify({
            "error": "Execution not found"
        }), 404
    
    execution.status = AgentStatus.ERROR
    execution.error = "Cancelled by user"
    execution.completed_at = __import__('time').time()
    
    return jsonify({
        "status": "cancelled",
        "execution_id": execution_id
    })


@app.route('/api/graphs/<paradigm>', methods=['GET'])
def get_graph_structure(paradigm: str):
    """Get the graph structure for a paradigm."""
    from graph_lib.paradigms import ParadigmFactory, ParadigmType
    
    try:
        paradigm_enum = ParadigmType(paradigm)
        graph = ParadigmFactory.create_paradigm(paradigm_enum)
        
        return jsonify({
            "paradigm": paradigm,
            "graph": graph
        })
    
    except ValueError:
        return jsonify({
            "error": f"Unknown paradigm: {paradigm}"
        }), 404


@app.route('/api/examples', methods=['GET'])
def list_examples():
    """List example agent configurations."""
    examples = [
        {
            "id": "research_agent",
            "name": "Research Agent",
            "description": "An agent that researches topics on the web",
            "paradigm": "react",
            "tools": ["search", "browser", "file_write"],
            "config": {
                "name": "Research Agent",
                "description": "An agent that researches topics and writes reports",
                "paradigm": "react",
                "tools": ["search", "browser", "file_write"],
                "model": "gpt-4",
                "max_iterations": 5
            }
        },
        {
            "id": "debate_team",
            "name": "Debate Team",
            "description": "Multiple agents debating a topic",
            "paradigm": "peer",
            "tools": ["search", "calculator"],
            "peers": ["proponent", "opponent", "moderator"],
            "config": {
                "name": "Debate Team",
                "description": "A team of agents that debate topics",
                "paradigm": "peer",
                "tools": ["search", "calculator"],
                "peers": ["proponent", "opponent", "moderator"],
                "model": "gpt-4",
                "max_iterations": 10
            }
        },
        {
            "id": "adaptive_solver",
            "name": "Adaptive Problem Solver",
            "description": "An agent that learns from its mistakes",
            "paradigm": "surprise",
            "tools": ["calculator", "code_execute"],
            "config": {
                "name": "Adaptive Problem Solver",
                "description": "An agent that adapts to unexpected outcomes",
                "paradigm": "surprise",
                "tools": ["calculator", "code_execute"],
                "model": "gpt-4",
                "max_iterations": 15
            }
        },
        {
            "id": "project_manager",
            "name": "Project Manager",
            "description": "Hierarchical agent for managing tasks",
            "paradigm": "hierarchical",
            "tools": ["calendar", "email", "database"],
            "workers": ["researcher", "developer", "tester"],
            "config": {
                "name": "Project Manager",
                "description": "A hierarchical agent for project management",
                "paradigm": "hierarchical",
                "tools": ["calendar", "email", "database"],
                "workers": ["researcher", "developer", "tester"],
                "model": "gpt-4",
                "max_iterations": 20
            }
        },
        {
            "id": "optimizer",
            "name": "Strategy Optimizer",
            "description": "Evolutionary agent that improves over time",
            "paradigm": "evolutionary",
            "tools": ["calculator", "api_call"],
            "config": {
                "name": "Strategy Optimizer",
                "description": "An agent that evolves its strategies",
                "paradigm": "evolutionary",
                "tools": ["calculator", "api_call"],
                "model": "gpt-4",
                "max_iterations": 50
            }
        }
    ]
    
    return jsonify({
        "examples": examples
    })


@app.route('/api/examples/<example_id>', methods=['GET'])
def get_example(example_id: str):
    """Get a specific example configuration."""
    examples = {
        "research_agent": {
            "name": "Research Agent",
            "description": "An agent that researches topics on the web",
            "paradigm": "react",
            "tools": ["search", "browser", "file_write"],
            "config": {
                "name": "Research Agent",
                "description": "An agent that researches topics and writes reports",
                "paradigm": "react",
                "tools": ["search", "browser", "file_write"],
                "model": "gpt-4",
                "max_iterations": 5
            }
        },
        "debate_team": {
            "name": "Debate Team",
            "description": "Multiple agents debating a topic",
            "paradigm": "peer",
            "tools": ["search", "calculator"],
            "peers": ["proponent", "opponent", "moderator"],
            "config": {
                "name": "Debate Team",
                "description": "A team of agents that debate topics",
                "paradigm": "peer",
                "tools": ["search", "calculator"],
                "peers": ["proponent", "opponent", "moderator"],
                "model": "gpt-4",
                "max_iterations": 10
            }
        }
    }
    
    example = examples.get(example_id)
    if not example:
        return jsonify({
            "error": "Example not found"
        }), 404
    
    return jsonify(example)


def _run_agent_background(execution_id: str, config: Dict[str, Any], input_data: Dict[str, Any]):
    """Run agent execution in background."""
    execution = agent_executions.get(execution_id)
    if not execution:
        return
    
    try:
        execution.status = AgentStatus.RUNNING
        
        # Simulate agent execution (in real implementation, this would run the LangGraph)
        import time
        time.sleep(2)  # Simulate work
        
        # Mock result
        execution.result = {
            "status": "completed",
            "output": f"Agent {config.get('name', 'unknown')} executed successfully",
            "input": input_data,
            "iterations": 5
        }
        execution.status = AgentStatus.COMPLETED
    
    except Exception as e:
        execution.error = str(e)
        execution.status = AgentStatus.ERROR
    
    finally:
        execution.completed_at = __import__('time').time()


def create_app():
    """Create and configure the Flask app."""
    return app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
