# ontological-playground-designer/tests/unit/test_world_compiler.py

import pytest
import os
import yaml
import json
import networkx as nx
import datetime
from unittest.mock import MagicMock

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.world_compiler import WorldCompiler
from src.core.rule_generator import GeneratedRule, GeneratedWorldRules
from src.utils.logger import setup_logging

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_world_compiler.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_sim_settings_path(tmp_path_factory):
    """Creates a temporary simulation_settings.yaml file."""
    config_dir = tmp_path_factory.mktemp("config_wc")
    path = config_dir / "simulation_settings.yaml"
    dummy_config = {
        'simulation_defaults': {
            'simulation_engine_version': "generic_test_sim_v1.0",
            'initial_world_size': {'x_dim': 50, 'y_dim': 50, 'z_dim': 1},
            'time_step_duration_ms': 50,
            'max_simulation_steps': 1000,
            'initial_agent_count': 10,
            'base_resource_regen_rate': 0.01,
            'output_format': "JSON"
        }
    }
    with open(path, 'w') as f:
        yaml.safe_dump(dummy_config, f)
    return str(path)

@pytest.fixture
def compiler_instance(dummy_sim_settings_path):
    """Returns a WorldCompiler instance."""
    return WorldCompiler(simulation_settings_path=dummy_sim_settings_path)

@pytest.fixture
def mock_generated_world_rules():
    """Returns a mock GeneratedWorldRules object."""
    world_name = "MockWorld"
    creation_time = datetime.datetime.now().isoformat()
    axioms_used = ["AXIOM_FLOURISH", "AXIOM_SUSTAIN"]

    rules = [
        GeneratedRule(
            id="Agent_Cooperation_Rule",
            description="Agents gain well-being from cooperative actions.",
            type="agent_behavior",
            parameters={"cooperation_reward_factor": 1.2},
            dependencies=["Agent_Spawner"],
            axiom_influence={"AXIOM_FLOURISH": 0.9}
        ),
        GeneratedRule(
            id="Resource_Regen_Law",
            description="Resource regeneration rate is tied to ecological health.",
            type="environmental_law",
            parameters={"regen_health_multiplier": 1.5},
            dependencies=["Resource_Source"],
            axiom_influence={"AXIOM_SUSTAIN": 0.8}
        ),
        GeneratedRule(
            id="Meta_Consistency_Check",
            description="All generated rules are checked for internal logical consistency.",
            type="meta_rule",
            parameters={"consistency_tolerance": 0.99},
            dependencies=["World_Root"],
            axiom_influence={"EPISTEMIC_COHERENCE_001": 1.0}
        ),
        GeneratedRule(
            id="Social_Sharing_Mechanism",
            description="Agents in high-resource areas share surplus.",
            type="social_dynamic",
            parameters={"sharing_threshold": 0.7},
            dependencies=["Agent_Spawner"],
            axiom_influence={"AXIOM_EQUITY": 0.7}
        )
    ]

    rule_graph = nx.DiGraph()
    rule_graph.add_node("Agent_Cooperation_Rule", type="agent_behavior")
    rule_graph.add_node("Resource_Regen_Law", type="environmental_law")
    rule_graph.add_edge("Agent_Cooperation_Rule", "Resource_Regen_Law", relation="positive_feedback")

    return GeneratedWorldRules(
        world_name=world_name,
        rules=rules,
        rule_graph=rule_graph,
        creation_timestamp=creation_time,
        axioms_used_ids=axioms_used,
        meta_data={"generator_version": "mock_v1.0"}
    )

# --- Test Cases ---

def test_world_compiler_initialization(compiler_instance):
    """Tests if the WorldCompiler initializes correctly and loads default settings."""
    assert compiler_instance is not None
    assert compiler_instance.simulation_settings['max_simulation_steps'] == 1000
    assert compiler_instance.simulation_settings['simulation_engine_version'] == "generic_test_sim_v1.0"
    logger.info("Test: WorldCompiler initializes correctly.")

def test_compile_world_produces_structured_config(compiler_instance, mock_generated_world_rules):
    """Tests if compile_world generates a well-structured configuration dictionary."""
    compiled_config = compiler_instance.compile_world(mock_generated_world_rules)

    assert isinstance(compiled_config, dict)
    assert 'world_metadata' in compiled_config
    assert compiled_config['world_metadata']['name'] == mock_generated_world_rules.world_name
    assert 'simulation_defaults' in compiled_config # Should contain the loaded defaults
    assert compiled_config['simulation_defaults']['initial_agent_count'] == 10 # From fixture
    assert 'generated_world_rules' in compiled_config
    
    # Check if rules are categorized correctly
    assert len(compiled_config['generated_world_rules']['agent_behaviors']) == 1
    assert compiled_config['generated_world_rules']['agent_behaviors'][0]['id'] == "Agent_Cooperation_Rule"
    assert len(compiled_config['generated_world_rules']['environmental_laws']) == 1
    assert len(compiled_config['generated_world_rules']['meta_rules']) == 1
    assert len(compiled_config['generated_world_rules']['social_dynamics']) == 1
    assert 'rule_interdependencies_graph' in compiled_config
    assert 'nodes' in compiled_config['rule_interdependencies_graph']
    assert 'edges' in compiled_config['rule_interdependencies_graph']
    assert len(compiled_config['rule_interdependencies_graph']['nodes']) == 2 # Rule_A, Rule_B
    assert len(compiled_config['rule_interdependencies_graph']['edges']) == 1
    
    logger.info("Test: compile_world produces structured config with correct categorization.")

def test_save_world_config_json_format(compiler_instance, mock_generated_world_rules, tmp_path):
    """Tests saving the compiled config in JSON format."""
    output_dir = tmp_path / "output_json"
    output_dir.mkdir()
    
    compiled_config = compiler_instance.compile_world(mock_generated_world_rules)
    compiler_instance.save_world_config(compiled_config, str(output_dir), format="json")

    saved_file = output_dir / f"{mock_generated_world_rules.world_name}.json"
    assert os.path.exists(saved_file)
    
    with open(saved_file, 'r') as f:
        loaded_config = json.load(f)
    assert loaded_config['world_metadata']['name'] == mock_generated_world_rules.world_name
    assert 'agent_behaviors' in loaded_config['generated_world_rules']
    logger.info("Test: Saving config in JSON format successful.")

def test_save_world_config_yaml_format(compiler_instance, mock_generated_world_rules, tmp_path):
    """Tests saving the compiled config in YAML format."""
    output_dir = tmp_path / "output_yaml"
    output_dir.mkdir()
    
    compiled_config = compiler_instance.compile_world(mock_generated_world_rules)
    compiler_instance.save_world_config(compiled_config, str(output_dir), format="yaml")

    saved_file = output_dir / f"{mock_generated_world_rules.world_name}.yaml"
    assert os.path.exists(saved_file)
    
    with open(saved_file, 'r') as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config['world_metadata']['name'] == mock_generated_world_rules.world_name
    assert 'environmental_laws' in loaded_config['generated_world_rules']
    logger.info("Test: Saving config in YAML format successful.")

def test_save_world_config_unsupported_format(compiler_instance, mock_generated_world_rules, tmp_path):
    """Tests error handling for an unsupported output format."""
    output_dir = tmp_path / "output_unsupported"
    output_dir.mkdir()
    
    compiled_config = compiler_instance.compile_world(mock_generated_world_rules)
    with pytest.raises(ValueError, match="Unsupported output format"):
        compiler_instance.save_world_config(compiled_config, str(output_dir), fo
