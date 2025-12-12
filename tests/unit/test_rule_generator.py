# ontological-playground-designer/tests/unit/test_rule_generator.py

import pytest
import os
import yaml
from unittest.mock import MagicMock, patch
import networkx as nx
import numpy as np
import torch # For mock model saving/loading

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.rule_generator import RuleGenerator, GeneratedRule, GeneratedWorldRules
from src.core.axiom_parser import AxiomSet, ParsedAxiom
from src.utils.logger import setup_logging

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_rule_generator.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_configs_path(tmp_path_factory):
    """Creates temporary config files (model_config.yaml, simulation_settings.yaml)."""
    config_dir = tmp_path_factory.mktemp("config_rg")
    
    # model_config.yaml
    model_config_path = config_dir / "model_config.yaml"
    model_config_content = {
        'rule_generator_model': {
            'type': "GraphTransformer",
            'architecture': {},
            'hyperparameters': {},
            'input_processing': {
                'axiom_embedding_model': "sentence-transformers/all-MiniLM-L6-v2"
            },
            'output_constraints': {
                'max_rule_complexity_score': 0.8,
                'rule_interdependency_threshold': 0.2,
                'diversity_penalty_coefficient': 0.05
            }
        }
    }
    with open(model_config_path, 'w') as f:
        yaml.safe_dump(model_config_content, f)
        
    # simulation_settings.yaml
    sim_settings_path = config_dir / "simulation_settings.yaml"
    sim_settings_content = {
        'simulation_defaults': {
            'initial_world_size': {'x_dim': 100, 'y_dim': 100},
            'max_simulation_steps': 1000
        }
    }
    with open(sim_settings_path, 'w') as f:
        yaml.safe_dump(sim_settings_content, f)

    return str(model_config_path), str(sim_settings_path)

@pytest.fixture
def rule_generator_instance(dummy_configs_path):
    """Returns a RuleGenerator instance."""
    model_path, sim_path = dummy_configs_path
    # Mock the actual model loading as we're testing the generator's logic, not the DL model itself
    with patch('src.core.rule_generator.RuleGenerator._load_generative_model_placeholder') as mock_load_model:
        mock_load_model.return_value = None # Ensure it doesn't try to load a real model
        rg = RuleGenerator(model_config_path=model_path, simulation_settings_path=sim_path)
        rg.generative_model = MagicMock(spec=torch.nn.Module) # Mock as if a model was loaded
        yield rg

@pytest.fixture
def mock_axiom_set():
    """Returns a mock AxiomSet for testing."""
    axioms = [
        ParsedAxiom(
            id="PHILOSOPHY_FLOURISHING_001",
            principle_text="Maximize long-term well-being.",
            priority=1,
            type="ethical",
            embedding=np.random.rand(128).tolist()
        ),
        ParsedAxiom(
            id="ECOLOGY_SUSTAINABILITY_001",
            principle_text="Ensure perpetual sustainability.",
            priority=2,
            type="environmental",
            embedding=np.random.rand(128).tolist()
        ),
        ParsedAxiom(
            id="EPISTEMIC_COHERENCE_001",
            principle_text="Maintain absolute logical coherence.",
            priority=0,
            type="foundational",
            embedding=np.random.rand(128).tolist()
        )
    ]
    return AxiomSet(axioms=axioms)

# --- Mock the _get_embedding_from_description for simulate_rule_generation_logic in ParadoxDetector
@pytest.fixture(autouse=True)
def mock_get_embedding_from_description():
    with patch('src.core.rule_generator.RuleGenerator._simulate_rule_generation_logic') as mock_simulate_logic:
        # Define a predictable output for the mock simulation logic
        def dummy_sim_logic(axiom_embeddings, axiom_ids, axiom_influence_weights, world_name):
            mock_rules = [
                GeneratedRule(id="Rule_A", description="Desc A", type="agent_behavior", parameters={}, axiom_influence={"AXIOM1":1.0}),
                GeneratedRule(id="Rule_B", description="Desc B", type="environmental_law", parameters={}, axiom_influence={"AXIOM2":1.0}),
            ]
            mock_graph = nx.DiGraph()
            mock_graph.add_node("Rule_A", type="agent_behavior")
            mock_graph.add_node("Rule_B", type="environmental_law")
            mock_graph.add_edge("Rule_A", "Rule_B", relation="influences")
            return mock_rules, mock_graph
        mock_simulate_logic.side_effect = dummy_sim_logic
        yield mock_simulate_logic

# --- Test Cases ---

def test_rule_generator_initialization(rule_generator_instance):
    """Tests if the RuleGenerator initializes correctly and loads configs."""
    assert rule_generator_instance is not None
    assert rule_generator_instance.model_config['type'] == "GraphTransformer"
    assert 'x_dim' in rule_generator_instance.sim_settings['initial_world_size']
    assert rule_generator_instance.generative_model is not None # Should be a mock now
    logger.info("Test: RuleGenerator initializes correctly.")

def test_generate_rules_produces_valid_output(rule_generator_instance, mock_axiom_set):
    """Tests if generate_rules produces a GeneratedWorldRules object with expected structure."""
    world_name = "TestWorld"
    world_rules = rule_generator_instance.generate_rules(mock_axiom_set, world_name)

    assert isinstance(world_rules, GeneratedWorldRules)
    assert world_rules.world_name == world_name
    assert isinstance(world_rules.rules, list)
    assert len(world_rules.rules) > 0 # Should have rules from mock_simulate_rule_generation_logic
    assert isinstance(world_rules.rule_graph, nx.DiGraph)
    assert world_rules.rule_graph.number_of_nodes() > 0
    assert isinstance(world_rules.creation_timestamp, str)
    assert world_rules.axioms_used_ids == [a.id for a in mock_axiom_set.axioms]
    assert 'rule_generator_model_type' in world_rules.meta_data
    logger.info("Test: generate_rules produces valid output structure.")

def test_axiom_influence_weighting(rule_generator_instance, mock_axiom_set):
    """Tests if axiom influence is correctly weighted based on priority."""
    weights = rule_generator_instance._apply_axiom_weighting(mock_axiom_set.axioms)
    
    # EPISTEMIC_COHERENCE_001 has priority 0 (highest)
    # PHILOSOPHY_FLOURISHING_001 has priority 1
    # ECOLOGY_SUSTAINABILITY_001 has priority 2
    
    # Expect axiom with priority 0 to have highest influence
    assert weights['EPISTEMIC_COHERENCE_001'] > weights['PHILOSOPHY_FLOURISHING_001']
    assert weights['PHILOSOPHY_FLOURISHING_001'] > weights['ECOLOGY_SUSTAINABILITY_001']
    
    # Check normalization (if applied in _apply_axiom_weighting)
    # For simplicity, checking relative order is sufficient for a unit test.
    logger.info("Test: Axiom influence weighting based on priority is correct.")

def test_handle_empty_axiom_set(rule_generator_instance, mock_axiom_set):
    """Tests generation with an empty axiom set."""
    empty_axiom_set = AxiomSet(axioms=[])
    world_name = "EmptyAxiomWorld"
    
    # Mock the internal logic to return empty if no axioms
    with patch('src.core.rule_generator.RuleGenerator._simulate_rule_generation_logic', return_value = ([], nx.DiGraph())) as mock_sim_logic:
        world_rules = rule_generator_instance.generate_rules(empty_axiom_set, world_name)
        assert len(world_rules.rules) == 0
        assert world_rules.rule_graph.number_of_nodes() == 0
        assert world_rules.axioms_used_ids == []
        logger.info("Test: Handles empty axiom set correctly.")

def test_missing_config_files(tmp_path):
    """Tests error handling when config files are missing."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        RuleGenerator(model_config_path=str(tmp_path / "non_existent_model.yaml"))
    
    model_path, _ = dummy_configs_path(tmp_path_factory=pytest.TempPathFactory())
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        RuleGenerator(model_config_path=model_path, simulation_settings_path=str(tmp_path / "non_existent_sim.yaml"))
    logger.info("Test: FileNotFoundError correctly raised for missing config files.")
