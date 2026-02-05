"""
Agent Paradigms for LangGraph-based Agent Engineer Architecture.

This module implements various agent collaboration paradigms:
- React: Standard think-act-observe loop
- Surprise: Self-improving based on unexpected outcomes
- Peer: Multi-agent collaboration/debate
- Hierarchical: Manager-worker agent structure
- Evolutionary: Agents that evolve over time
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4
import json
import random
from datetime import datetime

from .agents import BaseAgent, AgentConfig, AgentState, Message, MessageType


class ParadigmType(str, Enum):
    """Types of agent paradigms."""
    REACT = "react"
    SURPRISE = "surprise"
    PEER = "peer"
    HIERARCHICAL = "hierarchical"
    EVOLUTIONARY = "evolutionary"


@dataclass
class ParadigmConfig:
    """Configuration for an agent paradigm."""
    paradigm_type: ParadigmType = ParadigmType.REACT
    max_iterations: int = 10
    learning_rate: float = 0.1
    surprise_threshold: float = 0.7
    population_size: int = 5
    mutation_rate: float = 0.2
    elite_ratio: float = 0.1
    checkpoint_interval: int = 5


class ReactParadigm:
    """
    React Paradigm: Standard think-act-observe loop.
    
    This is the foundational paradigm where agents follow a cyclical
    process of reasoning, action, and observation.
    """
    
    @staticmethod
    def create_graph(config: AgentConfig) -> Dict[str, Any]:
        """Create a LangGraph for React paradigm."""
        return {
            "nodes": {
                "think": {
                    "type": "think",
                    "description": "Process context and plan actions",
                    "next": ["act"]
                },
                "act": {
                    "type": "act", 
                    "description": "Execute planned actions",
                    "next": ["observe"]
                },
                "observe": {
                    "type": "observe",
                    "description": "Process action results",
                    "next": ["think"]
                },
                "check_completion": {
                    "type": "condition",
                    "description": "Check if task is complete",
                    "next": {
                        "complete": ["end"],
                        "continue": ["think"]
                    }
                }
            },
            "edges": [
                ("think", "act"),
                ("act", "observe"),
                ("observe", "check_completion"),
                ("check_completion", "end")
            ],
            "entry_point": "think",
            "end_point": "end"
        }
    
    @staticmethod
    def create_node_functions(config: AgentConfig) -> Dict[str, Any]:
        """Create node functions for React paradigm."""
        return {
            "think": ReactParadigm._think_node,
            "act": ReactParadigm._act_node,
            "observe": ReactParadigm._observe_node,
            "check_completion": ReactParadigm._check_completion_node
        }
    
    @staticmethod
    async def _think_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Think node implementation."""
        return {
            "thought": "Analyzing current state and planning next steps",
            "plan": {"action": "continue"}
        }
    
    @staticmethod
    async def _act_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Act node implementation."""
        return {"action_result": "Action executed successfully"}
    
    @staticmethod
    async def _observe_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Observe node implementation."""
        return {"observation": "Result observed and processed"}
    
    @staticmethod
    async def _check_completion_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Check completion node implementation."""
        return {"status": "continue" if state.get("iteration", 0) < 10 else "complete"}


class SurpriseParadigm:
    """
    Surprise Paradigm: Self-improving based on unexpected outcomes.
    
    This paradigm enables agents to detect surprising events,
    learn from them, and improve their future behavior.
    """
    
    @staticmethod
    def create_graph(config: ParadigmConfig) -> Dict[str, Any]:
        """Create a LangGraph for Surprise paradigm."""
        return {
            "nodes": {
                "think": {
                    "type": "think",
                    "description": "Standard reasoning",
                    "next": ["act"]
                },
                "act": {
                    "type": "act",
                    "description": "Execute actions",
                    "next": ["observe"]
                },
                "observe": {
                    "type": "observe", 
                    "description": "Process results",
                    "next": ["detect_surprise"]
                },
                "detect_surprise": {
                    "type": "condition",
                    "description": "Detect if outcome was surprising",
                    "next": {
                        "surprising": ["analyze_surprise", "adapt"],
                        "expected": ["check_completion"]
                    }
                },
                "analyze_surprise": {
                    "type": "analyze",
                    "description": "Analyze surprising outcome",
                    "next": ["adapt"]
                },
                "adapt": {
                    "type": "adapt",
                    "description": "Adapt strategy based on surprise",
                    "next": ["check_completion"]
                },
                "check_completion": {
                    "type": "condition",
                    "description": "Check if task is complete",
                    "next": {
                        "complete": ["end"],
                        "continue": ["think"]
                    }
                }
            },
            "edges": [
                ("think", "act"),
                ("act", "observe"),
                ("observe", "detect_surprise"),
                ("detect_surprise", "analyze_surprise"),
                ("analyze_surprise", "adapt"),
                ("adapt", "check_completion"),
                ("detect_surprise", "check_completion"),
                ("check_completion", "end")
            ],
            "entry_point": "think",
            "end_point": "end"
        }
    
    @staticmethod
    def calculate_surprise(expected: Any, actual: Any) -> float:
        """
        Calculate surprise level based on expected vs actual outcome.
        
        Returns a value between 0 (no surprise) and 1 (complete surprise).
        """
        if expected == actual:
            return 0.0
        # Simplified surprise calculation
        return min(1.0, 0.5 + 0.5 * random.random())
    
    @staticmethod
    def adapt_strategy(surprise_analysis: Dict[str, Any], 
                      learning_rate: float) -> Dict[str, Any]:
        """
        Adapt strategy based on surprise analysis.
        
        This method updates the agent's internal model to improve
        future predictions and actions.
        """
        return {
            "updated_model": True,
            "learning_rate": learning_rate,
            "adjustments": surprise_analysis.get("insights", [])
        }


class PeerParadigm:
    """
    Peer Paradigm: Multi-agent collaboration and debate.
    
    This paradigm enables multiple agents to collaborate,
    debate ideas, and reach consensus or combine insights.
    """
    
    @staticmethod
    def create_graph(num_peers: int) -> Dict[str, Any]:
        """Create a LangGraph for Peer paradigm."""
        nodes = {
            "coordinator": {
                "type": "coordinator",
                "description": "Orchestrate peer discussion",
                "next": ["peer_discussion"]
            },
            "peer_discussion": {
                "type": "parallel",
                "description": "All peers contribute",
                "next": ["synthesize"]
            },
            "synthesize": {
                "type": "synthesize",
                "description": "Combine peer contributions",
                "next": ["validate"]
            },
            "validate": {
                "type": "condition",
                "description": "Validate synthesized result",
                "next": {
                    "valid": ["end"],
                    "invalid": ["refine"]
                }
            },
            "refine": {
                "type": "refine",
                "description": "Refine based on validation",
                "next": ["peer_discussion"]
            }
        }
        
        # Add peer nodes
        for i in range(num_peers):
            nodes[f"peer_{i}"] = {
                "type": "peer",
                "description": f"Peer {i} agent",
                "next": ["peer_discussion"]
            }
        
        edges = [
            ("coordinator", "peer_discussion"),
            ("synthesize", "validate"),
            ("validate", "end"),
            ("validate", "refine"),
            ("refine", "peer_discussion")
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "entry_point": "coordinator",
            "end_point": "end"
        }
    
    @staticmethod
    def create_debate_graph() -> Dict[str, Any]:
        """Create a LangGraph for debate-style collaboration."""
        return {
            "nodes": {
                "setup_debate": {
                    "type": "setup",
                    "description": "Set up debate topic and roles",
                    "next": ["opening_statements"]
                },
                "opening_statements": {
                    "type": "parallel",
                    "description": "All debaters present opening",
                    "next": ["cross_examination"]
                },
                "cross_examination": {
                    "type": "debate_round",
                    "description": "Cross-examination rounds",
                    "next": ["closing_statements"]
                },
                "closing_statements": {
                    "type": "parallel",
                    "description": "Final arguments",
                    "next": ["judge_deliberation"]
                },
                "judge_deliberation": {
                    "type": "judge",
                    "description": "Evaluate arguments and decide",
                    "next": ["end"]
                }
            },
            "edges": [
                ("setup_debate", "opening_statements"),
                ("opening_statements", "cross_examination"),
                ("cross_examination", "closing_statements"),
                ("closing_statements", "judge_deliberation"),
                ("judge_deliberation", "end")
            ],
            "entry_point": "setup_debate",
            "end_point": "end"
        }


class HierarchicalParadigm:
    """
    Hierarchical Paradigm: Manager-worker agent structure.
    
    This paradigm implements organizational structures where
    manager agents coordinate worker agents.
    """
    
    @staticmethod
    def create_graph(depth: int, workers_per_level: List[int]) -> Dict[str, Any]:
        """
        Create a hierarchical LangGraph.
        
        Args:
            depth: Number of management levels
            workers_per_level: Number of workers at each level
        """
        nodes = {}
        edges = []
        
        # Build hierarchical structure
        for level in range(depth):
            level_nodes = workers_per_level[level] if level < len(workers_per_level) else 1
            
            for worker_idx in range(level_nodes):
                node_id = f"level_{level}_worker_{worker_idx}"
                nodes[node_id] = {
                    "type": "worker" if level == depth - 1 else "manager",
                    "description": f"Level {level}, Worker {worker_idx}",
                    "level": level,
                    "next": []
                }
                
                # Connect to next level or end
                if level < depth - 1:
                    next_level_workers = workers_per_level[level + 1]
                    next_worker = (level + 1) * next_level_workers // level_nodes
                    nodes[node_id]["next"] = [f"level_{level + 1}_worker_{next_worker}"]
                else:
                    nodes[node_id]["next"] = ["supervisor_review"]
        
        nodes["supervisor_review"] = {
            "type": "review",
            "description": "Final review by supervisor",
            "next": ["end"]
        }
        
        edges = [
            (f"level_{l}_worker_{w}", nodes[f"level_{l}_worker_{w}"]["next"][0])
            for l in range(depth)
            for w in range(workers_per_level[l] if l < len(workers_per_level) else 1)
        ]
        edges.append(("supervisor_review", "end"))
        
        return {
            "nodes": nodes,
            "edges": edges,
            "entry_point": "level_0_worker_0",
            "end_point": "end"
        }
    
    @staticmethod
    def create_task_distribution_graph() -> Dict[str, Any]:
        """Create a graph for task distribution."""
        return {
            "nodes": {
                "task_reception": {
                    "type": "reception",
                    "description": "Receive and categorize task",
                    "next": ["task_analysis"]
                },
                "task_analysis": {
                    "type": "analysis",
                    "description": "Analyze task requirements",
                    "next": ["worker_selection"]
                },
                "worker_selection": {
                    "type": "selection",
                    "description": "Select optimal worker",
                    "next": ["task_delegation"]
                },
                "task_delegation": {
                    "type": "delegation",
                    "description": "Delegate task to worker",
                    "next": ["progress_monitoring"]
                },
                "progress_monitoring": {
                    "type": "monitoring",
                    "description": "Monitor worker progress",
                    "next": {
                        "in_progress": ["progress_monitoring"],
                        "completed": ["result_aggregation"],
                        "blocked": ["escalation"]
                    }
                },
                "result_aggregation": {
                    "type": "aggregation",
                    "description": "Aggregate results from workers",
                    "next": ["quality_check"]
                },
                "quality_check": {
                    "type": "condition",
                    "description": "Check result quality",
                    "next": {
                        "approved": ["end"],
                        "needs_revision": ["task_delegation"]
                    }
                },
                "escalation": {
                    "type": "escalation",
                    "description": "Escalate to higher level",
                    "next": ["task_delegation"]
                }
            },
            "edges": [
                ("task_reception", "task_analysis"),
                ("task_analysis", "worker_selection"),
                ("worker_selection", "task_delegation"),
                ("task_delegation", "progress_monitoring"),
                ("progress_monitoring", "result_aggregation"),
                ("result_aggregation", "quality_check"),
                ("quality_check", "end"),
                ("quality_check", "task_delegation"),
                ("progress_monitoring", "escalation"),
                ("escalation", "task_delegation")
            ],
            "entry_point": "task_reception",
            "end_point": "end"
        }


class EvolutionaryParadigm:
    """
    Evolutionary Paradigm: Agents that evolve over time.
    
    This paradigm implements evolutionary algorithms where
    agents evolve through selection, mutation, and crossover.
    """
    
    @staticmethod
    def create_graph() -> Dict[str, Any]:
        """Create a LangGraph for Evolutionary paradigm."""
        return {
            "nodes": {
                "initialize_population": {
                    "type": "init",
                    "description": "Create initial agent population",
                    "next": ["evaluate_fitness"]
                },
                "evaluate_fitness": {
                    "type": "evaluation",
                    "description": "Evaluate each agent's fitness",
                    "next": ["check_convergence"]
                },
                "check_convergence": {
                    "type": "condition",
                    "description": "Check if convergence criteria met",
                    "next": {
                        "converged": ["end"],
                        "continue": ["selection"]
                    }
                },
                "selection": {
                    "type": "selection",
                    "description": "Select best performing agents",
                    "next": ["crossover", "mutation"]
                },
                "crossover": {
                    "type": "crossover",
                    "description": "Cross over selected agents",
                    "next": ["mutation"]
                },
                "mutation": {
                    "type": "mutation",
                    "description": "Mutate offspring",
                    "next": ["replace_population"]
                },
                "replace_population": {
                    "type": "replace",
                    "description": "Replace old population with new",
                    "next": ["evaluate_fitness"]
                }
            },
            "edges": [
                ("initialize_population", "evaluate_fitness"),
                ("evaluate_fitness", "check_convergence"),
                ("check_convergence", "end"),
                ("check_convergence", "selection"),
                ("selection", "crossover"),
                ("selection", "mutation"),
                ("crossover", "mutation"),
                ("mutation", "replace_population"),
                ("replace_population", "evaluate_fitness")
            ],
            "entry_point": "initialize_population",
            "end_point": "end"
        }
    
    @staticmethod
    def create_individual(config: AgentConfig) -> Dict[str, Any]:
        """Create an individual agent for evolution."""
        return {
            "genome": {
                "system_prompt": config.system_prompt,
                "max_iterations": config.max_iterations,
                "tools": config.tools,
                "learning_rate": 0.1,
                "strategy_params": {}
            },
            "fitness": 0.0,
            "generation": 0,
            "id": str(uuid4())
        }
    
    @staticmethod
    def mutate(genome: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """Mutate a genome with given mutation rate."""
        mutated = genome.copy()
        
        if random.random() < mutation_rate:
            mutated["learning_rate"] *= random.uniform(0.8, 1.2)
        
        if random.random() < mutation_rate:
            # Mutate strategy parameters
            mutated["strategy_params"] = mutated.get("strategy_params", {})
            for key in mutated["strategy_params"]:
                mutated["strategy_params"][key] *= random.uniform(0.9, 1.1)
        
        return mutated
    
    @staticmethod
    def crossover(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two parent genomes."""
        child = {
            "learning_rate": (parent1.get("learning_rate", 0.1) + 
                            parent2.get("learning_rate", 0.1)) / 2,
            "strategy_params": parent1.get("strategy_params", {})
        }
        
        # Blend system prompts
        if random.random() < 0.5:
            child["system_prompt"] = parent1.get("system_prompt", "")
        else:
            child["system_prompt"] = parent2.get("system_prompt", "")
        
        return child
    
    @staticmethod
    def select_tournament(population: List[Dict[str, Any]], 
                         tournament_size: int) -> Dict[str, Any]:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.get("fitness", 0.0))


class ParadigmFactory:
    """Factory for creating paradigm graphs and configurations."""
    
    @staticmethod
    def create_paradigm(paradigm_type: ParadigmType, 
                       config: Optional[ParadigmConfig] = None) -> Dict[str, Any]:
        """Create a paradigm graph based on type."""
        config = config or ParadigmConfig()
        
        paradigm_maps = {
            ParadigmType.REACT: ReactParadigm.create_graph,
            ParadigmType.SURPRISE: SurpriseParadigm.create_graph,
            ParadigmType.PEER: lambda: PeerParadigm.create_graph(3),
            ParadigmType.HIERARCHICAL: lambda: HierarchicalParadigm.create_graph(2, [1, 3]),
            ParadigmType.EVOLUTIONARY: EvolutionaryParadigm.create_graph
        }
        
        creator = paradigm_maps.get(paradigm_type)
        if creator:
            return creator(config)
        raise ValueError(f"Unknown paradigm type: {paradigm_type}")
    
    @staticmethod
    def get_paradigm_description(paradigm_type: ParadigmType) -> str:
        """Get description of a paradigm."""
        descriptions = {
            ParadigmType.REACT: "Standard think-act-observe loop for reasoning",
            ParadigmType.SURPRISE: "Self-improving agent that learns from unexpected outcomes",
            ParadigmType.PEER: "Multi-agent collaboration through debate and synthesis",
            ParadigmType.HIERARCHICAL: "Manager-worker structure for task distribution",
            ParadigmType.EVOLUTIONARY: "Population-based evolution for continuous improvement"
        }
        return descriptions.get(paradigm_type, "Unknown paradigm")
    
    @staticmethod
    def list_paradigms() -> List[Dict[str, str]]:
        """List all available paradigms."""
        return [
            {"type": p.value, "description": ParadigmFactory.get_paradigm_description(p)}
            for p in ParadigmType
        ]
