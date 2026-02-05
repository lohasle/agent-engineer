# Agent Engineer Architecture

<p align="center">
  <strong>Build Professional AI Agents with Simple Configurations</strong>
</p>

<p align="center">
  A comprehensive LangGraph-based system for creating, deploying, and managing intelligent agents through simple configurations.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#paradigms">Paradigms</a> •
  <a href="#api">API</a> •
  <a href="#examples">Examples</a>
</p>

---

## ✨ Features

### Multi-Paradigm Support
- **React**: Standard think-act-observe loop for sequential reasoning
- **Surprise**: Self-improving agents that learn from unexpected outcomes
- **Peer**: Multi-agent collaboration through debate and synthesis
- **Hierarchical**: Manager-worker structures for complex task distribution
- **Evolutionary**: Population-based optimization for continuous improvement

### Dynamic Agent Generation
- Natural language agent descriptions
- Automatic LangGraph code generation
- Visual graph preview
- One-click deployment

### Professional Architecture
- Clean, maintainable code structure
- Comprehensive documentation
- Commercial-ready design patterns
- Extensible plugin system

---

## 🏗️ Architecture

```
agent-engineer/
├── frontend/                 # Web UI for agent creation
│   └── index.html           # Single-page application
├── backend/                  # API server
│   ├── server.py            # Flask API endpoints
│   └── generator.py         # LangGraph code generation
├── graph_lib/               # Core agent patterns
│   ├── agents.py            # Base agent classes
│   └── paradigms.py         # Paradigm implementations
├── examples/                # Sample configurations
│   ├── research_agent.json
│   ├── debate_team.json
│   ├── adaptive_solver.json
│   ├── project_manager.json
│   └── strategy_optimizer.json
└── README.md                # This file
```

### Component Details

#### Frontend (`frontend/`)
Clean, modern web interface for agent configuration:
- Paradigm selection with visual previews
- Tool selection and configuration
- Real-time graph visualization
- Code preview and download

#### Backend (`backend/`)
RESTful API for agent management:
- `/api/paradigms` - List available paradigms
- `/api/tools` - List available tools
- `/api/generate` - Generate LangGraph code
- `/api/execute` - Run agents
- `/api/graphs/<paradigm>` - Get graph structures

#### Graph Library (`graph_lib/`)
Core agent implementation:
- **agents.py**: BaseAgent, ManagerAgent, PeerAgent classes
- **paradigms.py**: React, Surprise, Peer, Hierarchical, Evolutionary patterns

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- OpenAI API key (or compatible LLM)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agent-engineer.git
cd agent-engineer

# Install backend dependencies
pip install -r backend/requirements.txt

# Start the backend
cd backend
python server.py

# Open frontend (serve the HTML file)
# Open frontend/index.html in your browser
```

### Basic Usage

```python
from backend.generator import generate_agent

# Define agent configuration
config = {
    "name": "My Research Agent",
    "description": "An agent that researches topics",
    "paradigm": "react",
    "tools": ["search", "browser", "file_write"],
    "model": "gpt-4",
    "max_iterations": 10
}

# Generate LangGraph code
files = generate_agent(config)

# Save generated code
for filename, content in files.items():
    with open(filename, 'w') as f:
        f.write(content)
```

---

## 🔧 Paradigms

### React Paradigm
Standard think-act-observe loop for sequential reasoning.

**Use cases:**
- Single-agent task completion
- Sequential information processing
- Simple automation workflows

**Graph structure:**
```
THINK → ACT → OBSERVE → (repeat or end)
```

### Surprise Paradigm
Self-improving agent that learns from unexpected outcomes.

**Use cases:**
- Adaptive learning systems
- Error recovery and resilience
- Strategy optimization

**Key features:**
- Surprise detection and analysis
- Automatic strategy adaptation
- Learning from mistakes

### Peer Paradigm
Multi-agent collaboration through debate and synthesis.

**Use cases:**
- Multi-perspective analysis
- Consensus building
- Collaborative problem-solving

**Key features:**
- Parallel peer processing
- Viewpoint synthesis
- Debate orchestration

### Hierarchical Paradigm
Manager-worker structure for task distribution.

**Use cases:**
- Complex workflow orchestration
- Task delegation
- Scalable processing

**Key features:**
- Task decomposition
- Worker selection and delegation
- Progress monitoring and aggregation

### Evolutionary Paradigm
Population-based evolution for continuous improvement.

**Use cases:**
- Strategy optimization
- Parameter tuning
- Adaptive systems

**Key features:**
- Tournament selection
- Crossover and mutation
- Fitness evaluation

---

## 📚 API Reference

### REST Endpoints

#### GET /api/paradigms
List all available paradigms.

```json
{
  "paradigms": [
    {
      "id": "react",
      "name": "React",
      "description": "Standard think-act-observe loop",
      "complexity": "low"
    }
  ]
}
```

#### POST /api/generate
Generate LangGraph code from configuration.

```json
{
  "name": "My Agent",
  "description": "An agent that...",
  "paradigm": "react",
  "tools": ["search", "browser"]
}
```

Response:
```json
{
  "status": "success",
  "agent_id": "uuid",
  "files": {
    "agent.py": "...",
    "graph.py": "...",
    "config.json": "...",
    "requirements.txt": "..."
  }
}
```

#### POST /api/execute
Run an agent with input data.

```json
{
  "config": {...},
  "input": {"task": "Your task"}
}
```

Response:
```json
{
  "execution_id": "uuid",
  "status": "started"
}
```

---

## 📦 Examples

### Research Agent
Simple React-based agent for topic research.

```bash
cd examples
# Load research_agent.json into the UI or API
```

### Debate Team
Multi-agent system with structured debate.

```json
{
  "paradigm": "peer",
  "peers": ["proponent", "opponent", "moderator"]
}
```

### Adaptive Solver
Self-improving agent with surprise learning.

```json
{
  "paradigm": "surprise",
  "learning_config": {
    "learning_rate": 0.1,
    "surprise_threshold": 0.7
  }
}
```

### Project Manager
Hierarchical agent for task management.

```json
{
  "paradigm": "hierarchical",
  "workers": ["researcher", "developer", "tester"]
}
```

---

## 🔒 Security

- Validate all inputs on the backend
- Use environment variables for API keys
- Implement rate limiting for API endpoints
- Sanitize generated code before execution

---

## 🧪 Testing

```bash
# Run backend tests
cd backend
python -m pytest tests/

# Run frontend tests (if using a framework)
cd frontend
npm test
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

- GitHub Issues: Report bugs and request features
- Documentation: See /docs for detailed guides
- Discord: Join our community server

---

<p align="center">
  Built with ❤️ using LangGraph
</p>
