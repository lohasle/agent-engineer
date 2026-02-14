"""Tests for Paradigms."""

import pytest
from graph_lib.paradigms import (
    ParadigmType,
    ParadigmConfig,
    ReactParadigm,
    SurpriseParadigm,
    PeerParadigm,
    HierarchicalParadigm,
    EvolutionaryParadigm,
    ParadigmFactory,
)


class TestParadigmType:
    def test_paradigm_types(self):
        assert ParadigmType.REACT == "react"
        assert ParadigmType.SURPRISE == "surprise"
        assert ParadigmType.PEER == "peer"
        assert ParadigmType.HIERARCHICAL == "hierarchical"
        assert ParadigmType.EVOLUTIONARY == "evolutionary"


class TestParadigmConfig:
    def test_default_config(self):
        config = ParadigmConfig()
        assert config.paradigm_type == ParadigmType.REACT
        assert config.max_iterations == 10
        assert config.learning_rate == 0.1

    def test_custom_config(self):
        config = ParadigmConfig(
            paradigm_type=ParadigmType.SURPRISE,
            max_iterations=20,
            learning_rate=0.05,
        )
        assert config.paradigm_type == ParadigmType.SURPRISE
        assert config.max_iterations == 20
        assert config.learning_rate == 0.05


class TestReactParadigm:
    def test_create_graph(self):
        from graph_lib.agents import AgentConfig
        config = AgentConfig(name="Test")
        graph = ReactParadigm.create_graph(config)
        
        assert "nodes" in graph
        assert "edges" in graph
        assert "entry_point" in graph
        assert graph["entry_point"] == "think"


class TestSurpriseParadigm:
    def test_calculate_surprise(self):
        surprise = SurpriseParadigm.calculate_surprise("expected", "actual")
        assert 0.0 <= surprise <= 1.0

    def test_no_surprise(self):
        surprise = SurpriseParadigm.calculate_surprise("same", "same")
        assert surprise == 0.0


class TestPeerParadigm:
    def test_create_graph(self):
        graph = PeerParadigm.create_graph(3)
        
        assert "nodes" in graph
        assert "coordinator" in graph["nodes"]
        assert graph["entry_point"] == "coordinator"

    def test_create_debate_graph(self):
        graph = PeerParadigm.create_debate_graph()
        
        assert "nodes" in graph
        assert "judge_deliberation" in graph["nodes"]


class TestHierarchicalParadigm:
    def test_create_graph(self):
        graph = HierarchicalParadigm.create_graph(2, [1, 3])
        
        assert "nodes" in graph
        assert "supervisor_review" in graph["nodes"]

    def test_create_task_distribution_graph(self):
        graph = HierarchicalParadigm.create_task_distribution_graph()
        
        assert "nodes" in graph
        assert "task_reception" in graph["nodes"]


class TestEvolutionaryParadigm:
    def test_create_graph(self):
        graph = EvolutionaryParadigm.create_graph()
        
        assert "nodes" in graph
        assert "initialize_population" in graph["nodes"]

    def test_mutate(self):
        genome = {"learning_rate": 0.1}
        mutated = EvolutionaryParadigm.mutate(genome, 0.5)
        assert "learning_rate" in mutated

    def test_crossover(self):
        parent1 = {"learning_rate": 0.1, "system_prompt": "parent1"}
        parent2 = {"learning_rate": 0.2, "system_prompt": "parent2"}
        child = EvolutionaryParadigm.crossover(parent1, parent2)
        assert 0.1 <= child["learning_rate"] <= 0.2


class TestParadigmFactory:
    def test_create_react_paradigm(self):
        graph = ParadigmFactory.create_paradigm(ParadigmType.REACT)
        assert "nodes" in graph

    def test_get_paradigm_description(self):
        desc = ParadigmFactory.get_paradigm_description(ParadigmType.REACT)
        assert "think-act-observe" in desc.lower()

    def test_list_paradigms(self):
        paradigms = ParadigmFactory.list_paradigms()
        assert len(paradigms) == 5
