# ontological-playground-designer/tests/unit/test_flourishing_evaluator.py

import pytest
import os
import yaml
import json
import networkx as nx
import numpy as np
import datetime
from unittest.mock import MagicMock, patch

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.flourishing_evaluator import FlourishingEvaluator, EvaluationMetric, WorldEvaluationReport
from src.core.axiom_parser import AxiomSet, ParsedAxiom
from src.core.rule_generator import GeneratedRule, GeneratedWorldRules
from src.utils.logger import setup_logging

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_flourishing_evaluator.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_model_and_axiom_configs_path(tmp_path_factory):
    """Creates temporary model_config.yaml and axioms.yaml files."""
    config_dir = tmp_path_factory.mktemp("config_fe")
    
    # model_config.yaml
    model_config_path = config_dir / "model_config.yaml"
    model_config_content = {
        'flourishing_evaluator_model': {
            'type': "TimeDistributedGraphCNN",
            'architecture': {},
            'hyperparameters': {},
            'training_data_generation': {
                'simulation_duration_steps': 1000
            }
        },
        'paradox_detector_model': { # Also needed for mock rule_generator
            'type': "GraphAttentionNetwork", 'architecture': {}, 'hyperparameters': {},
            'detection_logic': {}
        }
    }
    with open(model_config_path, 'w') as f:
        yaml.safe_dump(model_config_content, f)
        
    # axioms.yaml
    axioms_path = config_dir / "axioms.yaml"
    axioms_content = {
        'world_axioms': [
            {'id': 'PHILOSOPHY_FLOURISHING_001', 'principle': 'Maximize long-term well-being.', 'priority': 1, 'type': 'ethical'},
            {'id': 'ECOLOGY_SUSTAINABILITY_001', 'principle': 'Ensure perpetual sustainability.', 'priority': 2, 'type': 'environmental'},
            {'id': 'SOCIAL_EQUITY_001', 'principle': 'Minimize disparities.', 'priority': 3, 'type': 'social'},
            {'id': 'EPISTEMIC_COHERENCE_001', 'principle': 'Maintain logical consistency.', 'priority': 0, 'type': 'foundational'},
            {'id': 'ETHICS_AGENCY_001', 'principle': 'Protect agent autonomy.', 'priority': 1, 'type': 'ethical'}
        ]
    }
    with open(axioms_path, 'w') as f:
        yaml.safe_dump(axioms_content, f)

    return str(model_config_path), str(axioms_path)

@pytest.fixture
def evaluator_instance(dummy_model_and_axiom_configs_path):
    """Returns a FlourishingEvaluator instance."""
    model_path, axioms_path = dummy_model_and_axiom_configs_path
    with patch('src.core.flourishing_evaluator.FlourishingEvaluator._load_evaluator_model_placeholder') as mock_load_model:
        mock_load_model.return_value = None # Ensure it doesn't try to load a real model
        fe = FlourishingEvaluator(model_config_path=model_path, axioms_config_path=axioms_path)
        fe.evaluator_model = MagicMock(spec=torch.nn.Module) # Mock as if a model was loaded
        yield fe

@pytest.fixture
def mock_axiom_set():
    """Returns a mock AxiomSet for testing."""
    axioms = [
        ParsedAxiom(id="PHILOSOPHY_FLOURISHING_001", principle_text="Maximize long-term well-being.", priority=1, type="ethical", embedding=np.random.rand(128).tolist()),
        ParsedAxiom(id="ECOLOGY_SUSTAINABILITY_001", principle_text="Ensure perpetual sustainability.", priority=2, type="environmental", embedding=np.random.rand(128).tolist()),
        ParsedAxiom(id="EPISTEMIC_COHERENCE_001", principle_text="Maintain logical consistency.", priority=0, type="foundational", embedding=np.random.rand(128).tolist())
    ]
    return AxiomSet(axioms=axioms)

@pytest.fixture
def mock_compiled_config():
    """Returns a mock compiled world configuration dictionary."""
    world_name = "MockEvaluatedWorld"
    creation_time = datetime.datetime.now().isoformat()
    axioms_used = ["PHILOSOPHY_FLOURISHING_001", "ECOLOGY_SUSTAINABILITY_001", "EPISTEMIC_COHERENCE_001"]

    mock_rules_data = [
        GeneratedRule(id="Agent_Cooperation", description="Agents cooperate to gain well-being.", type="agent_behavior", parameters={"cooperation_reward_factor": 0.8}, axiom_influence={"PHILOSOPHY_FLOURISHING_001":0.9}),
        GeneratedRule(id="Resource_Regen", description="Resources regenerate based on ecological health.", type="environmental_law", parameters={"regen_health_multiplier": 1.1}, axiom_influence={"ECOLOGY_SUSTAINABILITY_001":0.8}),
        GeneratedRule(id="Meta_Consistency", description="All rules must be logically consistent.", type="meta_rule", parameters={"consistency_tolerance": 0.99}, axiom_influence={"EPISTEMIC_COHERENCE_001":1.0})
    ]
    
    mock_graph = nx.DiGraph()
    mock_graph.add_node("Agent_Cooperation")
    mock_graph.add_node("Resource_Regen")
    mock_graph.add_edge("Agent_Cooperation", "Resource_Regen", relation="affects")

    return {
        'world_metadata': {
            'name': world_name,
            'creation_timestamp': creation_time,
            'axioms_influencing_design': axioms_used,
            'designed_by': "Mock AI"
        },
        'simulation_defaults': {
            'initial_agent_count': 100,
            'base_resource_regen_rate': 0.01,
            'initial_agent_cooperation_tendency': 0.5,
            'environmental_diversity_index': 0.7,
            'initial_wealth_distribution_model': 'uniform'
        },
        'generated_world_rules': {
            'agent_behaviors': [r.__dict__ for r in mock_rules_data if r.type == 'agent_behavior'],
            'environmental_laws': [r.__dict__ for r in mock_rules_data if r.type == 'environmental_law'],
            'meta_rules': [r.__dict__ for r in mock_rules_data if r.type == 'meta_rule'],
            'social_dynamics': [],
            'resource_mechanics': [],
            'system_mechanics': [],
            'unclassified_rules': []
        },
        'rule_interdependencies_graph': {
            'nodes': [{'id': node_id, 'attributes': data} for node_id, data in mock_graph.nodes(data=True)],
            'edges': [{'source': u, 'target': v, 'attributes': data} for u, v, data in mock_graph.edges(data=True)]
        }
    }


# --- Mock the _simulate_prediction_logic for evaluation ---
@pytest.fixture(autouse=True)
def mock_simulate_prediction_logic():
    with patch('src.core.flourishing_evaluator.FlourishingEvaluator._simulate_prediction_logic') as mock_sim_logic:
        # Define a predictable output for the mock prediction logic
        def dummy_prediction_logic(features):
            return {
                'total_flourishing_score': 0.85,
                'sustainability_index': 0.90,
                'equity_distribution': 0.70,
                'agency_protection_score': 0.80,
                'resilience_score': 0.75,
            }
        mock_sim_logic.side_effect = dummy_prediction_logic
        yield mock_sim_logic

# --- Test Cases ---

def test_flourishing_evaluator_initialization(evaluator_instance):
    """Tests if the FlourishingEvaluator initializes correctly and loads configs."""
    assert evaluator_instance is not None
    assert 'type' in evaluator_instance.model_config
    assert len(evaluator_instance.axioms_config) > 0 # Should have loaded axioms
    assert evaluator_instance.evaluator_model is not None # Should be a mock now
    logger.info("Test: FlourishingEvaluator initializes correctly.")

def test_extract_features_from_world_config(evaluator_instance, mock_compiled_config):
    """Tests if key features are correctly extracted from the compiled config."""
    features = evaluator_instance._extract_features_from_world_config(mock_compiled_config)
    
    assert isinstance(features, dict)
    assert features['initial_agent_count'] == 100
    assert features['base_resource_regen_rate'] == 0.01
    assert features['num_agent_behaviors'] == 1
    assert features['rule_graph_density'] > 0 # Based on mock graph
    assert features['aggregate_cooperation_reward'] > 0
    logger.info("Test: Features extracted correctly from world config.")

def test_evaluate_world_produces_valid_report(evaluator_instance, mock_compiled_config, mock_axiom_set):
    """Tests if evaluate_world produces a WorldEvaluationReport with expected structure."""
    paradox_risk = 0.1 # Simulate a low paradox risk
    report = evaluator_instance.evaluate_world(mock_compiled_config, mock_axiom_set, paradox_risk)

    assert isinstance(report, WorldEvaluationReport)
    assert report.world_name == mock_compiled_config['world_metadata']['name']
    assert isinstance(report.evaluation_timestamp, str)
    assert report.overall_flourishing_score > 0
    assert report.paradox_risk_score == paradox_risk
    assert len(report.predicted_metrics) > 0
    assert len(report.axiom_adherence_scores) > 0
    assert isinstance(report.recommendations, list)
    logger.info("Test: evaluate_world produces valid report structure.")

def test_axiom_adherence_scores_mapping(evaluator_instance, mock_compiled_config, mock_axiom_set):
    """Tests if predicted metrics are correctly mapped to axiom adherence scores."""
    report = evaluator_instance.evaluate_world(mock_compiled_config, mock_axiom_set)

    # Check mapping for specific axioms based on mock_simulate_prediction_logic
    assert report.axiom_adherence_scores['PHILOSOPHY_FLOURISHING_001'] == pytest.approx(0.85)
    assert report.axiom_adherence_scores['ECOLOGY_SUSTAINABILITY_001'] == pytest.approx(0.90)
    assert report.axiom_adherence_scores['EPISTEMIC_COHERENCE_001'] == pytest.approx(1.0 - report.paradox_risk_score) # From 1 - paradox_risk for demo
    logger.info("Test: Axiom adherence scores are correctly mapped.")

def test_generate_recommendations(evaluator_instance, mock_compiled_config, mock_axiom_set):
    """Tests if recommendations are generated based on adherence scores."""
    # Simulate low adherence to trigger specific recommendations
    with patch('src.core.flourishing_evaluator.FlourishingEvaluator._simulate_prediction_logic') as mock_sim_logic:
        mock_sim_logic.return_value = {
            'total_flourishing_score': 0.4, # Low
            'sustainability_index': 0.5,    # Low
            'equity_distribution': 0.8,
            'paradox_risk_score_input': 0.05 # Low paradox risk
        }
        report = evaluator_instance.evaluate_world(mock_compiled_config, mock_axiom_set)
        
        assert "Detected areas for improvement" in report.recommendations[0]
        assert "PHILOSOPHY_FLOURISHING_001" in report.recommendations[1]
        assert "ECOLOGY_SUSTAINABILITY_001" in report.recommendations[3]
        logger.info("Test: Recommendations generated for low adherence scores.")

    # Test for high adherence resulting in positive feedback
    with patch('src.core.flourishing_evaluator.FlourishingEvaluator._simulate_prediction_logic') as mock_sim_logic:
        mock_sim_logic.return_value = {
            'total_flourishing_score': 0.9,
            'sustainability_index': 0.95,
            'equity_distribution': 0.9,
            'paradox_risk_score_input': 0.01
        }
        report = evaluator_instance.evaluate_world(mock_compiled_config, mock_axiom_set, paradox_risk_score=0.01)
        assert "strong axiomatic alignment" in report.recommendations[0]
        logger.info("Test: Positive recommendations generated for high adherence scores.")

def test_missing_config_files(tmp_path):
    """Tests error handling when config files are missing."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        FlourishingEvaluator(model_config_path=str(tmp_path / "non_existent_model_fe.yaml"))
    
    model_path, _ = dummy_model_and_axiom_configs_path(tmp_path_factory=pytest.TempPathFactory())
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        FlourishingEvaluator(model_config_path=model_path, axioms_config_path=str(tmp_path / "non_existent_axioms_fe.yaml"))
    logger.info("Test: FileNotFoundError correctly raised for missing config files.")
