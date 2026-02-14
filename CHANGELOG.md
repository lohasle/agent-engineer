# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-15

### Added
- Multi-paradigm agent support (React, Surprise, Peer, Hierarchical, Evolutionary)
- LangGraph code generation from natural language configurations
- RESTful API backend with Flask
- Web-based frontend for agent configuration
- Comprehensive documentation (English & Chinese)
- CI/CD pipeline with GitHub Actions
- Example agent configurations

### Core Features
- **React Paradigm**: Standard think-act-observe loop for sequential reasoning
- **Surprise Paradigm**: Self-improving agents that learn from unexpected outcomes
- **Peer Paradigm**: Multi-agent collaboration through debate and synthesis
- **Hierarchical Paradigm**: Manager-worker structures for complex task distribution
- **Evolutionary Paradigm**: Population-based optimization for continuous improvement

### API Endpoints
- `GET /api/health` - Health check endpoint
- `GET /api/paradigms` - List available paradigms
- `GET /api/tools` - List available tools
- `POST /api/generate` - Generate LangGraph code
- `POST /api/execute` - Execute agents
- `GET /api/graphs/<paradigm>` - Get graph structures

## [0.1.0] - 2026-02-06

### Added
- Initial project structure
- Basic agent classes and paradigms
- Code generator module
