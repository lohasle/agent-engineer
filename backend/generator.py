"""
LangGraph Code Generator for Agent Engineer Architecture.

This module generates LangGraph code from user configurations,
supporting dynamic agent construction and multiple paradigms.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4
import json
import textwrap


@dataclass
class AgentTemplate:
    """Template for agent code generation."""
    name: str
    description: str
    paradigm: str
    tools: List[str] = field(default_factory=list)
    model: str = "gpt-4"
    system_prompt: str = ""
    max_iterations: int = 10
    workers: List[str] = field(default_factory=list)
    peers: List[str] = field(default_factory=list)


class LangGraphGenerator:
    """Generator for LangGraph code from agent configurations."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load code templates."""
        return {
            "react_agent": self._react_agent_template(),
            "surprise_agent": self._surprise_agent_template(),
            "peer_agent": self._peer_agent_template(),
            "hierarchical_agent": self._hierarchical_agent_template(),
            "evolutionary_agent": self._evolutionary_agent_template(),
            "graph_builder": self._graph_builder_template(),
            "node_functions": self._node_functions_template()
        }
    
    def generate_from_config(self, config: AgentTemplate) -> Dict[str, str]:
        """Generate LangGraph code from agent configuration."""
        paradigm_templates = {
            "react": "react_agent",
            "surprise": "surprise_agent",
            "peer": "peer_agent",
            "hierarchical": "hierarchical_agent",
            "evolutionary": "evolutionary_agent"
        }
        
        template_name = paradigm_templates.get(config.paradigm, "react_agent")
        base_code = self._generate_base_code(config)
        graph_code = self._generate_graph_code(config)
        
        return {
            "agent": base_code,
            "graph": graph_code,
            "config": self._generate_config_json(config),
            "requirements": self._generate_requirements(config)
        }
    
    def _generate_base_code(self, config: AgentTemplate) -> str:
        """Generate base agent code."""
        template = self.templates["react_agent"]
        return template.format(
            agent_name=config.name,
            description=config.description,
            model=config.model,
            max_iterations=config.max_iterations,
            tools=", ".join(f'"{t}"' for t in config.tools),
            system_prompt=config.system_prompt or f"You are {config.name}, {config.description}."
        )
    
    def _generate_graph_code(self, config: AgentTemplate) -> str:
        """Generate graph construction code."""
        template = self.templates["graph_builder"]
        return template.format(
            agent_name=config.name,
            paradigm=config.paradigm
        )
    
    def _generate_config_json(self, config: AgentTemplate) -> str:
        """Generate configuration JSON."""
        config_dict = {
            "name": config.name,
            "description": config.description,
            "paradigm": config.paradigm,
            "model": config.model,
            "max_iterations": config.max_iterations,
            "tools": config.tools,
            "system_prompt": config.system_prompt,
            "workers": config.workers,
            "peers": config.peers
        }
        return json.dumps(config_dict, indent=2)
    
    def _generate_requirements(self, config: AgentTemplate) -> str:
        """Generate requirements.txt content."""
        requirements = [
            "langgraph>=0.0.20",
            "langchain>=0.1.0",
            "openai>=1.0.0",
            "pydantic>=2.0.0",
            "python-dotenv>=1.0.0"
        ]
        
        if "search" in config.tools or "web_search" in config.tools:
            requirements.append("brave-search>=1.0.0")
        
        if "browser" in config.tools:
            requirements.append("playwright>=1.40.0")
        
        return "\n".join(requirements)
    
    def _react_agent_template(self) -> str:
        """React agent template."""
        return textwrap.dedent('''
            """
            React Agent: {agent_name}
            
            {description}
            
            This agent follows the think-act-observe pattern using LangGraph.
            """
            
            from typing import Any, Dict, List, TypedDict
            from langgraph.graph import StateGraph, END
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            
            # Define state schema
            class AgentState(TypedDict):
                messages: List[Dict[str, Any]]
                context: Dict[str, Any]
                plan: Dict[str, Any]
                result: Dict[str, Any]
                iteration: int
            
            
            # Initialize LLM
            llm = ChatOpenAI(model="{model}", temperature=0.7)
            
            
            # Node functions
            def think(state: AgentState) -> AgentState:
                """Think node: Process context and plan actions."""
                messages = state.get("messages", [])
                context = state.get("context", {{}})
                
                # Build thinking prompt
                prompt = f"""
                You are thinking about the current task.
                Context: {{context}}
                Previous messages: {{messages[-3:]}}
                
                What should be your next action?
                """
                
                response = llm.invoke([
                    SystemMessage(content="You are a thoughtful reasoning agent."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "plan": {{
                        "thought": response.content,
                        "action": "continue"
                    }},
                    "messages": messages + [{{"role": "assistant", "content": response.content}}]
                }}
            
            
            def act(state: AgentState) -> AgentState:
                """Act node: Execute planned actions."""
                plan = state.get("plan", {{}})
                context = state.get("context", {{}})
                
                # Execute action based on plan
                action_result = {{
                    "action": plan.get("action"),
                    "result": "Action executed successfully",
                    "context": context
                }}
                
                return {{
                    **state,
                    "result": action_result,
                    "iteration": state.get("iteration", 0) + 1
                }}
            
            
            def observe(state: AgentState) -> AgentState:
                """Observe node: Process action results."""
                result = state.get("result", {{}})
                context = state.get("context", {{}})
                
                # Update context based on observation
                updated_context = {{
                    **context,
                    "last_result": result,
                    "observation_count": context.get("observation_count", 0) + 1
                }}
                
                return {{
                    **state,
                    "context": updated_context
                }}
            
            
            def check_completion(state: AgentState) -> str:
                """Check if task is complete."""
                iteration = state.get("iteration", 0)
                context = state.get("context", {{}})
                
                if iteration >= {max_iterations}:
                    return "complete"
                
                # Add your completion criteria here
                if context.get("task_complete"):
                    return "complete"
                
                return "continue"
            
            
            # Build the graph
            def build_graph() -> StateGraph:
                """Build the React agent graph."""
                workflow = StateGraph(AgentState)
                
                # Add nodes
                workflow.add_node("think", think)
                workflow.add_node("act", act)
                workflow.add_node("observe", observe)
                workflow.add_node("check_completion", check_completion)
                
                # Add edges
                workflow.set_entry_point("think")
                workflow.add_edge("think", "act")
                workflow.add_edge("act", "observe")
                workflow.add_edge("observe", "check_completion")
                
                # Conditional edge from check_completion
                workflow.add_conditional_edges(
                    "check_completion",
                    check_completion,
                    {{
                        "continue": "think",
                        "complete": END
                    }}
                )
                
                return workflow.compile()
            
            
            # Execute the agent
            async def run_agent(input_context: Dict[str, Any]) -> Dict[str, Any]:
                """Run the React agent with given input."""
                graph = build_graph()
                
                initial_state = {{
                    "messages": [],
                    "context": input_context,
                    "plan": {{}},
                    "result": {{}},
                    "iteration": 0
                }}
                
                final_state = await graph.ainvoke(initial_state)
                return final_state
            
            
            if __name__ == "__main__":
                import asyncio
                result = asyncio.run(run_agent({{"task": "Your task here"}}))
                print(result)
        ''').strip()
    
    def _surprise_agent_template(self) -> str:
        """Surprise agent template."""
        return textwrap.dedent('''
            """
            Surprise Agent: {agent_name}
            
            {description}
            
            This agent learns from unexpected outcomes using LangGraph.
            """
            
            from typing import Any, Dict, List, TypedDict
            from langgraph.graph import StateGraph, END
            from langchain_openai import ChatOpenAI
            import random
            
            
            class AgentState(TypedDict):
                messages: List[Dict[str, Any]]
                context: Dict[str, Any]
                expected_outcome: Any
                actual_outcome: Any
                surprise_level: float
                learned_lessons: List[str]
                iteration: int
            
            
            llm = ChatOpenAI(model="{model}", temperature=0.7)
            
            
            def think(state: AgentState) -> AgentState:
                """Think and predict expected outcome."""
                context = state.get("context", {{}})
                
                prompt = f"""
                Given the current context, predict what outcome you expect:
                Context: {{context}}
                
                Predict the expected outcome and your confidence (0-1).
                """
                
                response = llm.invoke([
                    SystemMessage(content="You predict expected outcomes."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "expected_outcome": response.content,
                    "messages": state.get("messages", []) + [{{"role": "assistant", "content": response.content}}]
                }}
            
            
            def act(state: AgentState) -> AgentState:
                """Execute action and observe actual outcome."""
                context = state.get("context", {{}})
                
                # Simulate action execution
                actual_outcome = f"Actual result based on {{context}}"
                
                return {{
                    **state,
                    "actual_outcome": actual_outcome,
                    "iteration": state.get("iteration", 0) + 1
                }}
            
            
            def observe(state: AgentState) -> AgentState:
                """Observe and calculate surprise level."""
                expected = state.get("expected_outcome", "")
                actual = state.get("actual_outcome", "")
                
                # Calculate surprise
                surprise = self._calculate_surprise(expected, actual)
                
                return {{
                    **state,
                    "surprise_level": surprise,
                    "context": {{
                        **state.get("context", {{}}),
                        "surprise_detected": surprise > 0.7
                    }}
                }}
            
            
            def detect_surprise(state: AgentState) -> str:
                """Detect if outcome was surprising."""
                if state.get("surprise_level", 0) > 0.7:
                    return "surprising"
                return "expected"
            
            
            def analyze_surprise(state: AgentState) -> AgentState:
                """Analyze why outcome was surprising."""
                expected = state.get("expected_outcome", "")
                actual = state.get("actual_outcome", "")
                
                prompt = f"""
                Analyze why this outcome was surprising:
                Expected: {{expected}}
                Actual: {{actual}}
                
                What did you learn?
                """
                
                response = llm.invoke([
                    SystemMessage(content="You analyze surprising outcomes."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "learned_lessons": state.get("learned_lessons", []) + [response.content],
                    "context": {{
                        **state.get("context", {{}}),
                        "last_lesson": response.content
                    }}
                }}
            
            
            def adapt(state: AgentState) -> AgentState:
                """Adapt strategy based on surprise."""
                lessons = state.get("learned_lessons", [])
                
                prompt = f"""
                Based on these lessons, adapt your strategy:
                {{lessons}}
                
                How will you change your approach?
                """
                
                response = llm.invoke([
                    SystemMessage(content="You adapt based on learning."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "context": {{
                        **state.get("context", {{}}),
                        "adapted_strategy": response.content
                    }}
                }}
            
            
            def _calculate_surprise(self, expected: str, actual: str) -> float:
                """Calculate surprise level (simplified)."""
                if expected.lower() == actual.lower():
                    return 0.0
                return min(1.0, 0.5 + 0.5 * random.random())
            
            
            def build_graph() -> StateGraph:
                """Build the Surprise agent graph."""
                workflow = StateGraph(AgentState)
                
                workflow.add_node("think", think)
                workflow.add_node("act", act)
                workflow.add_node("observe", observe)
                workflow.add_node("detect_surprise", detect_surprise)
                workflow.add_node("analyze_surprise", analyze_surprise)
                workflow.add_node("adapt", adapt)
                
                workflow.set_entry_point("think")
                workflow.add_edge("think", "act")
                workflow.add_edge("act", "observe")
                workflow.add_edge("observe", "detect_surprise")
                
                workflow.add_conditional_edges(
                    "detect_surprise",
                    detect_surprise,
                    {{
                        "surprising": ["analyze_surprise", "adapt"],
                        "expected": ["think"]
                    }}
                )
                
                workflow.add_edge("analyze_surprise", "adapt")
                workflow.add_edge("adapt", "think")
                
                return workflow.compile()
            
            
            async def run_agent(input_context: Dict[str, Any]) -> Dict[str, Any]:
                """Run the Surprise agent."""
                graph = build_graph()
                initial_state = {{
                    "messages": [],
                    "context": input_context,
                    "expected_outcome": None,
                    "actual_outcome": None,
                    "surprise_level": 0.0,
                    "learned_lessons": [],
                    "iteration": 0
                }}
                return await graph.ainvoke(initial_state)
        ''').strip()
    
    def _peer_agent_template(self) -> str:
        """Peer collaboration agent template."""
        return textwrap.dedent('''
            """
            Peer Agent: {agent_name}
            
            {description}
            
            This agent collaborates with peers using LangGraph.
            """
            
            from typing import Any, Dict, List, TypedDict
            from langgraph.graph import StateGraph, END
            from langgraph.constants import Send
            from langchain_openai import ChatOpenAI
            
            
            class AgentState(TypedDict):
                task: str
                peer_contributions: List[Dict[str, Any]]
                synthesized_result: str
                validation_result: bool
                iteration: int
            
            
            llm = ChatOpenAI(model="{model}", temperature=0.7)
            
            
            def coordinator(state: AgentState) -> AgentState:
                """Coordinate peer discussion."""
                return {{
                    **state,
                    "peer_contributions": [],
                    "iteration": state.get("iteration", 0) + 1
                }}
            
            
            def peer_contribution(peer_id: str) -> Dict[str, Any]:
                """Create a peer contribution node."""
                def contribution_node(state: AgentState) -> AgentState:
                    prompt = f"""
                    Peer {{peer_id}} contributing to task: {{state['task']}}
                    
                    Provide your perspective and insights.
                    """
                    
                    response = llm.invoke([
                        SystemMessage(content=f"You are peer {{peer_id}}."),
                        HumanMessage(content=prompt)
                    ])
                    
                    return {{
                        **state,
                        "peer_contributions": state.get("peer_contributions", []) + [{{
                            "peer_id": peer_id,
                            "contribution": response.content
                        }}]
                    }}
                return contribution_node
            
            
            def peer_discussion(state: AgentState) -> List[Dict[str, Any]]:
                """Spawn peer discussion."""
                peers = {config.peers if config.peers else ['peer_1', 'peer_2', 'peer_3']}
                return [Send(peer, {{"task": state["task"]}}) for peer in peers]
            
            
            def synthesize(state: AgentState) -> AgentState:
                """Synthesize peer contributions."""
                contributions = state.get("peer_contributions", [])
                
                prompt = f"""
                Synthesize these peer contributions into a unified result:
                {{contributions}}
                
                Provide a coherent synthesis.
                """
                
                response = llm.invoke([
                    SystemMessage(content="You synthesize diverse perspectives."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "synthesized_result": response.content
                }}
            
            
            def validate(state: AgentState) -> str:
                """Validate synthesized result."""
                result = state.get("synthesized_result", "")
                # Simple validation
                return "valid" if len(result) > 10 else "invalid"
            
            
            def refine(state: AgentState) -> AgentState:
                """Refine based on validation feedback."""
                prompt = f"""
                Refine this result based on validation feedback:
                Current: {{state['synthesized_result']}}
                
                Make improvements.
                """
                
                response = llm.invoke([
                    SystemMessage(content="You refine outputs."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "synthesized_result": response.content
                }}
            
            
            def build_graph() -> StateGraph:
                """Build the Peer agent graph."""
                workflow = StateGraph(AgentState)
                
                workflow.add_node("coordinator", coordinator)
                workflow.add_node("synthesize", synthesize)
                workflow.add_node("validate", validate)
                workflow.add_node("refine", refine)
                
                # Add peer nodes
                peers = {config.peers if config.peers else ['peer_1', 'peer_2', 'peer_3']}
                for peer in peers:
                    workflow.add_node(peer, peer_contribution(peer))
                
                workflow.set_entry_point("coordinator")
                workflow.add_edge("coordinator", "peer_discussion")
                
                # Use Send for parallel peer processing
                workflow.add_conditional_edges(
                    "coordinator",
                    peer_discussion,
                    {peer: "synthesize" for peer in peers}
                )
                
                workflow.add_edge("synthesize", "validate")
                
                workflow.add_conditional_edges(
                    "validate",
                    validate,
                    {{"valid": END, "invalid": "refine"}}
                )
                
                workflow.add_edge("refine", "synthesize")
                
                return workflow.compile()
            
            
            async def run_agent(task: str) -> Dict[str, Any]:
                """Run the Peer agent."""
                graph = build_graph()
                initial_state = {{
                    "task": task,
                    "peer_contributions": [],
                    "synthesized_result": "",
                    "validation_result": True,
                    "iteration": 0
                }}
                return await graph.ainvoke(initial_state)
        ''').strip()
    
    def _hierarchical_agent_template(self) -> str:
        """Hierarchical agent template."""
        return textwrap.dedent('''
            """
            Hierarchical Agent: {agent_name}
            
            {description}
            
            This agent uses manager-worker structure via LangGraph.
            """
            
            from typing import Any, Dict, List, TypedDict
            from langgraph.graph import StateGraph, END
            from langchain_openai import ChatOpenAI
            
            
            class AgentState(TypedDict):
                task: str
                subtasks: List[Dict[str, Any]]
                worker_results: Dict[str, Any]
                current_worker: str
                quality_score: float
                iteration: int
            
            
            llm = ChatOpenAI(model="{model}", temperature=0.7)
            
            
            def task_reception(state: AgentState) -> AgentState:
                """Receive and categorize task."""
                return {{
                    **state,
                    "subtasks": [],
                    "worker_results": {{}}
                }}
            
            
            def task_analysis(state: AgentState) -> AgentState:
                """Analyze task requirements."""
                task = state.get("task", "")
                
                prompt = f"""
                Analyze this task and break it down:
                Task: {{task}}
                
                List subtasks needed.
                """
                
                response = llm.invoke([
                    SystemMessage(content="You decompose tasks."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "subtasks": [{{"description": response.content, "status": "pending"}}]
                }}
            
            
            def worker_selection(state: AgentState) -> AgentState:
                """Select optimal worker for task."""
                subtasks = state.get("subtasks", [])
                
                # Simple worker selection logic
                workers = {config.workers if config.workers else ['worker_1', 'worker_2', 'worker_3']}
                
                return {{
                    **state,
                    "current_worker": workers[0],
                    "subtasks": subtasks
                }}
            
            
            def task_delegation(state: AgentState) -> AgentState:
                """Delegate task to worker."""
                worker = state.get("current_worker", "")
                subtasks = state.get("subtasks", [])
                
                # Mark current subtask as in progress
                for subtask in subtasks:
                    if subtask.get("status") == "pending":
                        subtask["status"] = "in_progress"
                        subtask["worker"] = worker
                        break
                
                return {{
                    **state,
                    "subtasks": subtasks
                }}
            
            
            def worker_execution(state: AgentState) -> AgentState:
                """Execute task by worker."""
                worker = state.get("current_worker", "")
                subtasks = state.get("subtasks", {})
                
                # Simulate worker execution
                result = f"Result from {{worker}}"
                
                return {{
                    **state,
                    "worker_results": {{
                        **state.get("worker_results", {{}}),
                        worker: result
                    }}
                }}
            
            
            def progress_monitoring(state: AgentState) -> str:
                """Monitor worker progress."""
                subtasks = state.get("subtasks", [])
                
                all_complete = all(s.get("status") == "completed" for s in subtasks)
                any_blocked = any(s.get("status") == "blocked" for s in subtasks)
                
                if all_complete:
                    return "completed"
                elif any_blocked:
                    return "blocked"
                return "in_progress"
            
            
            def result_aggregation(state: AgentState) -> AgentState:
                """Aggregate results from workers."""
                results = state.get("worker_results", {})
                
                prompt = f"""
                Aggregate these worker results:
                {{results}}
                
                Provide a unified summary.
                """
                
                response = llm.invoke([
                    SystemMessage(content="You aggregate results."),
                    HumanMessage(content=prompt)
                ])
                
                return {{
                    **state,
                    "aggregated_result": response.content
                }}
            
            
            def quality_check(state: AgentState) -> str:
                """Check result quality."""
                result = state.get("aggregated_result", "")
                return "approved" if len(result) > 20 else "needs_revision"
            
            
            def build_graph() -> StateGraph:
                """Build the Hierarchical agent graph."""
                workflow = StateGraph(AgentState)
                
                workflow.add_node("task_reception", task_reception)
                workflow.add_node("task_analysis", task_analysis)
                workflow.add_node("worker_selection", worker_selection)
                workflow.add_node("task_delegation", task_delegation)
                workflow.add_node("worker_execution", worker_execution)
                workflow.add_node("progress_monitoring", progress_monitoring)
                workflow.add_node("result_aggregation", result_aggregation)
                workflow.add_node("quality_check", quality_check)
                
                workflow.set_entry_point("task_reception")
                workflow.add_edge("task_reception", "task_analysis")
                workflow.add_edge("task_analysis", "worker_selection")
                workflow.add_edge("worker_selection", "task_delegation")
                workflow.add_edge("task_delegation", "worker_execution")
                workflow.add_edge("worker_execution", "progress_monitoring")
                
                workflow.add_conditional_edges(
                    "progress_monitoring",
                    progress_monitoring,
                    {{
                        "in_progress": "worker_execution",
                        "completed": "result_aggregation",
                        "blocked": "worker_selection"
                    }}
                )
                
                workflow.add_edge("result_aggregation", "quality_check")
                
                workflow.add_conditional_edges(
                    "quality_check",
                    quality_check,
                    {{"approved": END, "needs_revision": "task_delegation"}}
                )
                
                return workflow.compile()
            
            
            async def run_agent(task: str) -> Dict[str, Any]:
                """Run the Hierarchical agent."""
                graph = build_graph()
                initial_state = {{
                    "task": task,
                    "subtasks": [],
                    "worker_results": {{}},
                    "current_worker": "",
                    "quality_score": 0.0,
                    "iteration": 0
                }}
                return await graph.ainvoke(initial_state)
        ''').strip()
    
    def _evolutionary_agent_template(self) -> str:
        """Evolutionary agent template."""
        return textwrap.dedent('''
            """
            Evolutionary Agent: {agent_name}
            
            {description}
            
            This agent evolves through selection, mutation, and crossover.
            """
            
            from typing import Any, Dict, List, TypedDict
            from langgraph.graph import StateGraph, END
            from langchain_openai import ChatOpenAI
            import random
            
            
            class AgentState(TypedDict):
                population: List[Dict[str, Any]]
                generation: int
                best_individual: Dict[str, Any]
                convergence_score: float
                task_results: List[Dict[str, Any]]
            
            
            llm = ChatOpenAI(model="{model}", temperature=0.7)
            
            
            def initialize_population(state: AgentState) -> AgentState:
                """Create initial agent population."""
                population_size = 5
                population = []
                
                for i in range(population_size):
                    individual = {{
                        "id": f"gen{{0}}__{{i}}",
                        "genome": {{
                            "system_prompt": "You are an evolving agent.",
                            "strategy": random.choice(["aggressive", "conservative", "balanced"]),
                            "temperature": random.uniform(0.1, 1.0)
                        }},
                        "fitness": 0.0
                    }}
                    population.append(individual)
                
                return {{
                    **state,
                    "population": population,
                    "generation": 0
                }}
            
            
            def evaluate_fitness(state: AgentState) -> AgentState:
                """Evaluate each agent's fitness."""
                population = state.get("population", [])
                
                for individual in population:
                    # Evaluate fitness based on task results
                    individual["fitness"] = random.uniform(0.5, 1.0)
                
                # Find best individual
                best = max(population, key=lambda x: x.get("fitness", 0))
                
                return {{
                    **state,
                    "population": population,
                    "best_individual": best,
                    "convergence_score": best.get("fitness", 0)
                }}
            
            
            def check_convergence(state: AgentState) -> str:
                """Check if convergence criteria met."""
                if state.get("convergence_score", 0) > 0.95:
                    return "converged"
                if state.get("generation", 0) >= 50:
                    return "converged"
                return "continue"
            
            
            def selection(state: AgentState) -> AgentState:
                """Select best performing agents."""
                population = state.get("population", [])
                elite_ratio = 0.1
                
                # Tournament selection
                selected = []
                for _ in range(len(population)):
                    tournament = random.sample(population, 3)
                    winner = max(tournament, key=lambda x: x.get("fitness", 0))
                    selected.append(winner)
                
                return {{
                    **state,
                    "selected": selected
                }}
            
            
            def crossover(state: AgentState) -> AgentState:
                """Cross over selected agents."""
                selected = state.get("selected", [])
                offspring = []
                
                for i in range(0, len(selected) - 1, 2):
                    parent1 = selected[i]
                    parent2 = selected[i + 1]
                    
                    child = {{
                        "id": f"gen{{state.get('generation', 0)}}__{{len(offspring)}}",
                        "genome": {{
                            "strategy": parent1.get("genome", {{}}).get("strategy"),
                            "temperature": (parent1.get("genome", {{}}).get("temperature", 0.5) + 
                                          parent2.get("genome", {{}}).get("temperature", 0.5)) / 2
                        }},
                        "fitness": 0.0
                    }}
                    offspring.append(child)
                
                return {{
                    **state,
                    "offspring": offspring
                }}
            
            
            def mutation(state: AgentState) -> AgentState:
                """Mutate offspring."""
                offspring = state.get("offspring", [])
                mutation_rate = 0.2
                
                for individual in offspring:
                    if random.random() < mutation_rate:
                        individual["genome"]["temperature"] *= random.uniform(0.8, 1.2)
                    if random.random() < mutation_rate:
                        individual["genome"]["strategy"] = random.choice(["aggressive", "conservative", "balanced"])
                
                return {{
                    **state,
                    "mutated_offspring": offspring
                }}
            
            
            def replace_population(state: AgentState) -> AgentState:
                """Replace old population with new generation."""
                old_population = state.get("population", [])
                offspring = state.get("mutated_offspring", [])
                
                # Elitism: keep best from old population
                sorted_pop = sorted(old_population, key=lambda x: x.get("fitness", 0), reverse=True)
                elite_count = max(1, len(old_population) // 10)
                elites = sorted_pop[:elite_count]
                
                # Create new population
                new_population = elites + offspring[:len(old_population) - elite_count]
                
                # Update generation
                for i, ind in enumerate(new_population):
                    ind["id"] = f"gen{{state.get('generation', 0) + 1}}__{{i}}"
                
                return {{
                    **state,
                    "population": new_population,
                    "generation": state.get("generation", 0) + 1
                }}
            
            
            def build_graph() -> StateGraph:
                """Build the Evolutionary agent graph."""
                workflow = StateGraph(AgentState)
                
                workflow.add_node("initialize_population", initialize_population)
                workflow.add_node("evaluate_fitness", evaluate_fitness)
                workflow.add_node("check_convergence", check_convergence)
                workflow.add_node("selection", selection)
                workflow.add_node("crossover", crossover)
                workflow.add_node("mutation", mutation)
                workflow.add_node("replace_population", replace_population)
                
                workflow.set_entry_point("initialize_population")
                workflow.add_edge("initialize_population", "evaluate_fitness")
                workflow.add_edge("evaluate_fitness", "check_convergence")
                
                workflow.add_conditional_edges(
                    "check_convergence",
                    check_convergence,
                    {{"converged": END, "continue": "selection"}}
                )
                
                workflow.add_edge("selection", "crossover")
                workflow.add_edge("crossover", "mutation")
                workflow.add_edge("mutation", "replace_population")
                workflow.add_edge("replace_population", "evaluate_fitness")
                
                return workflow.compile()
            
            
            async def run_agent() -> Dict[str, Any]:
                """Run the Evolutionary agent."""
                graph = build_graph()
                initial_state = {{
                    "population": [],
                    "generation": 0,
                    "best_individual": None,
                    "convergence_score": 0.0,
                    "task_results": []
                }}
                return await graph.ainvoke(initial_state)
        ''').strip()
    
    def _graph_builder_template(self) -> str:
        """Graph builder template."""
        return textwrap.dedent('''
            """
            Graph Builder for {agent_name}
            
            This module provides the LangGraph builder for the {paradigm} paradigm.
            """
            
            from .agent import build_graph, run_agent
            
            
            def get_graph():
                """Get the compiled graph."""
                return build_graph()
            
            
            async def execute(input_data):
                """Execute the agent with input data."""
                return await run_agent(input_data)
            
            
            if __name__ == "__main__":
                import asyncio
                result = asyncio.run(execute({{"task": "Your task"}}))
                print(result)
        ''').strip()
    
    def _node_functions_template(self) -> str:
        """Node functions template."""
        return textwrap.dedent('''
            """
            Node Functions for {agent_name}
            
            This module contains the node functions used in the agent graph.
            """
            
            from typing import Any, Dict
            
            
            # Node function implementations
            # Add your custom node functions here
            
            def custom_node(state: Dict[str, Any]) -> Dict[str, Any]:
                """Custom node implementation."""
                return {{
                    **state,
                    "custom_value": "processed"
                }}
        ''').strip()


def generate_agent(agent_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate LangGraph code from agent configuration dictionary.
    
    Args:
        agent_config: Dictionary containing agent configuration
        
    Returns:
        Dictionary with generated code files
    """
    config = AgentTemplate(
        name=agent_config.get("name", "unnamed_agent"),
        description=agent_config.get("description", "An agent"),
        paradigm=agent_config.get("paradigm", "react"),
        tools=agent_config.get("tools", []),
        model=agent_config.get("model", "gpt-4"),
        system_prompt=agent_config.get("system_prompt", ""),
        max_iterations=agent_config.get("max_iterations", 10),
        workers=agent_config.get("workers", []),
        peers=agent_config.get("peers", [])
    )
    
    generator = LangGraphGenerator()
    return generator.generate_from_config(config)


def save_generated_code(files: Dict[str, str], output_dir: str) -> None:
    """Save generated code to files."""
    import os
    
    for filename, content in files.items():
        filepath = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
