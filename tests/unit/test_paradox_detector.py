# ontological-playground-designer/tests/unit/test_paradox_detector.py

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

from src.core.paradox_detector import ParadoxDetector, DetectedParadox, ParadoxDetectionReport
from src.core.axiom_parser import AxiomSet, ParsedAxiom
from src.core.rule_generator import GeneratedRule, GeneratedWorldRules
from src.utils.logger import setup_logging
from src.utils.math_utils import cosine_similarity # Use real math_utils for semantic test

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_paradox_detector.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_model_and_axiom_configs_path_pd(tmp_path_factory):
    """Creates temporary model_config.yaml and axioms.yaml files for ParadoxDetector."""
    config_dir = tmp_path_factory.mktemp("config_pd")
    
    # model_config.yaml
    model_config_path = config_dir / "model_config.yaml"
    model_config_content = {
        'paradox_detector_model': {
            'type': "GraphAttentionNetwork",
            'architecture': {},
            'hyperparameters': {},
            'detection_logic': {
                'rule_embedding_threshold': 0.85, # Set a clear threshold for semantic conflict
                'ethical_conflict_threshold': 0.6, # Set a clear threshold for ethical conflict
                'logical_form_parser': "Z3_SMT_solver_adapter"
            }
        }
    }
    with open(model_config_path, 'w') as f:
        yaml.safe_dump(model_config_content, f)
        
    # axioms.yaml
    axioms_path = config_dir / "axioms.yaml"
    axioms_content = {
        'world_axioms': [
            {'id': 'PHILOSOPHY_FLOURISHING_001', 'principle': 'Maximize long-term well-being and adaptive capacity of all sentient agents.', 'priority': 1, 'type': 'ethical', 'embedding': np.random.rand(128).tolist()},
            {'id': 'ECOLOGY_SUSTAINABILITY_001', 'principle': 'Ensure perpetual resource sustainability and regeneration.', 'priority': 2, 'type': 'environmental', 'embedding': np.random.rand(128).tolist()},
            {'id': 'EPISTEMIC_COHERENCE_001', 'principle': 'Maintain absolute logical and conceptual coherence.', 'priority': 0, 'type': 'foundational', 'embedding': np.random.rand(128).tolist()},
            {'id': 'ETHICS_AGENCY_001', 'principle': 'Protect agent autonomy and subjective well-being.', 'priority': 1, 'type': 'ethical', 'embedding': np.random.rand(128).tolist()}
        ]
    }
    with open(axioms_path, 'w') as f:
        yaml.safe_dump(axioms_content, f)

    return str(model_config_path), str(axioms_path)

@pytest.fixture
def detector_instance(dummy_model_and_axiom_configs_path_pd):
    """Returns a ParadoxDetector instance."""
    model_path, axioms_path = dummy_model_and_axiom_configs_path_pd
    with patch('src.core.paradox_detector.ParadoxDetector._load_paradox_model_placeholder') as mock_load_model:
        mock_load_model.return_value = None # Ensure it doesn't try to load a real model
        pd = ParadoxDetector(model_config_path=model_path, axioms_config_path=axioms_path)
        pd.paradox_detector_model = MagicMock(spec=torch.nn.Module) # Mock as if a model was loaded
        yield pd

@pytest.fixture
def mock_axiom_set_pd(dummy_model_and_axiom_configs_path_pd):
    """Returns a mock AxiomSet for ParadoxDetector, with explicit embeddings."""
    _, axioms_path = dummy_model_and_axiom_configs_path_pd
    with open(axioms_path, 'r') as f:
        axioms_content = yaml.safe_load(f)['world_axioms']
    
    # Manually create ParsedAxiom objects with actual (random) embeddings
    parsed_axioms = [
        ParsedAxiom(id=a['id'], principle_text=a['principle'], priority=a['priority'], type=a['type'], embedding=np.random.rand(128).tolist())
        for a in axioms_content
    ]
    return AxiomSet(axioms=parsed_axioms)

@pytest.fixture
def mock_generated_world_rules_pd(mock_axiom_set_pd):
    """Returns a mock GeneratedWorldRules object, including rules with specific conflicts."""
    world_name = "ParadoxTestWorld"
    creation_time = datetime.datetime.now().isoformat()
    axioms_used = [a.id for a in mock_axiom_set_pd.axioms]

    rules = [
        GeneratedRule( # Rule that semantically conflicts with itself (mocked similar embeddings)
            id="Rule_HighEfficiency",
            description="Agents must prioritize collective resource extraction efficiency.",
            type="agent_behavior",
            parameters={"efficiency_target": 0.95},
            dependencies=[],
            axiom_influence={"PHILOSOPHY_FLOURISHING_001": 0.3, "EPISTEMIC_COHERENCE_001": 0.9} # Low flourishing influence
        ),
        GeneratedRule( # Rule semantically similar to Rule_HighEfficiency, but different implication
            id="Rule_IndividualWellbeing",
            description="Agents must ensure maximum individual well-being at all costs.",
            type="agent_behavior",
            parameters={"individual_wellbeing_floor": 0.8},
            dependencies=[],
            axiom_influence={"PHILOSOPHY_FLOURISHING_001": 0.9, "ETHICS_AGENCY_001": 0.9} # High flourishing influence
        ),
        GeneratedRule( # A neutral rule
            id="Rule_EnvironmentalScan",
            description="Agents periodically scan the environment for resource nodes.",
            type="system_mechanic",
            parameters={"scan_frequency": 10},
            dependencies=[],
            axiom_influence={"ECOLOGY_SUSTAINABILITY_001": 0.7}
        )
    ]
    
    # Mock embeddings for these rules to simulate semantic conflict
    # These will be used by the mocked _get_embedding_from_description in _simulate_paradox_detection_logic
    with patch('src.core.paradox_detector.ParadoxDetector._get_embedding_from_description') as mock_get_rule_emb:
        # Generate predictable, high-similarity embeddings for the conflicting rules
        mock_get_rule_emb.side_effect = lambda desc: {
            "Agents must prioritize collective resource extraction efficiency.": np.array([0.9, 0.1, 0.1, 0.1]).tolist(),
            "Agents must ensure maximum individual well-being at all costs.": np.array([0.88, 0.12, 0.08, 0.11]).tolist(), # High similarity
            "Agents periodically scan the environment for resource nodes.": np.array([0.1, 0.9, 0.1, 0.1]).tolist(),
        }.get(desc, np.random.rand(128).tolist()) # Default to random if not found
        
        # Now create the actual GeneratedWorldRules
        world_rules = GeneratedWorldRules(
            world_name=world_name,
            rules=rules,
            rule_graph=nx.DiGraph(), # Mock rule_graph is fine for this level of mock
            creation_timestamp=creation_time,
            axioms_used_ids=axioms_used,
            meta_data={"generator_version": "mock_v1.0"}
        )
        yield world_rules

# --- Test Cases ---

def test_paradox_detector_initialization(detector_instance):
    """Tests if the ParadoxDetector initializes correctly and loads configs."""
    assert detector_instance is not None
    assert 'type' in detector_instance.model_config
    assert len(detector_instance.axioms_config_data) > 0 # Should have loaded axioms
    assert detector_instance.paradox_detector_model is not None # Should be a mock now
    logger.info("Test: ParadoxDetector initializes correctly.")

def test_detect_paradoxes_semantic_conflict(detector_instance, mock_generated_world_rules, mock_axiom_set_pd):
    """Tests if semantic conflicts between rules are detected."""
    # We've set up rules in mock_generated_world_rules to have high semantic similarity
    # and conflict potential, _simulate_paradox_detection_logic should pick this up.
    report = detector_instance.detect_paradoxes(mock_generated_world_rules, mock_axiom_set_pd)

    assert isinstance(report, ParadoxDetectionReport)
    assert report.world_name == mock_generated_world_rules.world_name
    assert len(report.detected_paradoxes) >= 1 # Expecting at least one semantic conflict
    
    semantic_paradox_found = False
    for p in report.detected_paradoxes:
        if p.type == "semantic_inconsistency":
            semantic_paradox_found = True
            assert "Rule_HighEfficiency" in p.involved_rules_ids
            assert "Rule_IndividualWellbeing" in p.involved_rules_ids
            assert p.severity > 0.8 # Should be high due to high mock similarity
            logger.info(f"Detected semantic paradox: {p.description}")
    assert semantic_paradox_found, "Expected a semantic inconsistency paradox but none found."
    logger.info("Test: Semantic conflicts between rules are correctly detected.")

def test_detect_paradoxes_ethical_tension(detector_instance, mock_generated_world_rules, mock_axiom_set_pd):
    """Tests if ethical conflicts (low axiom influence vs. high priority axiom) are detected."""
    # Rule_HighEfficiency has low influence for flourishing axiom (0.3)
    # This should trigger an ethical_tension paradox
    report = detector_instance.detect_paradoxes(mock_generated_world_rules, mock_axiom_set_pd)

    ethical_paradox_found = False
    for p in report.detected_paradoxes:
        if p.type == "ethical_tension":
            ethical_paradox_found = True
            assert "Rule_HighEfficiency" in p.involved_rules_ids
            assert "PHILOSOPHY_FLOURISHING_001" in p.involved_rules_ids
            assert p.severity > 0.5 # Should be moderate to high
            logger.info(f"Detected ethical tension: {p.description}")
    assert ethical_paradox_found, "Expected an ethical tension paradox but none found."
    logger.info("Test: Ethical conflicts are correctly detected.")

def test_detect_paradoxes_no_paradoxes(detector_instance, mock_axiom_set_pd):
    """Tests detection when no paradoxes are present (mocked behavior)."""
    # Use a clean set of rules
    rules = [
        GeneratedRule(id="Rule_Good", description="Promotes well-being.", type="agent_behavior", parameters={}, axiom_influence={"PHILOSOPHY_FLOURISHING_001":1.0}),
        GeneratedRule(id="Rule_Clean", description="Ensures clean environment.", type="environmental_law", parameters={}, axiom_influence={"ECOLOGY_SUSTAINABILITY_001":1.0}),
    ]
    clean_world_rules = GeneratedWorldRules(
        world_name="CleanWorld", rules=rules, rule_graph=nx.DiGraph(), creation_timestamp=datetime.datetime.now().isoformat(), axioms_used_ids=[]
    )
    # Mock _simulate_paradox_detection_logic to return empty list
    with patch('src.core.paradox_detector.ParadoxDetector._simulate_paradox_detection_logic', return_value=[]):
        report = detector_instance.detect_paradoxes(clean_world_rules, mock_axiom_set_pd)
        assert len(report.detected_paradoxes) == 0
        assert report.total_paradox_risk_score == 0.0
        logger.info("Test: No paradoxes detected for clean rule set.")

def test_total_paradox_risk_score_calculation(detector_instance, mock_generated_world_rules, mock_axiom_set_pd):
    """Tests if the total paradox risk score is correctly aggregated."""
    report = detector_instance.detect_paradoxes(mock_generated_world_rules, mock_axiom_set_pd)
    
    assert report.total_paradox_risk_score >= 0.0
    assert report.total_paradox_risk_score <= 1.0 # Score should be normalized
    # Basic check: if there are paradoxes, score should be > 0
    assert report.total_paradox_risk_score > 0
    logger.info("Test: Total paradox risk score calculated correctly.")

def test_missing_config_files(tmp_path):
    """Tests error handling when config files are missing."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        ParadoxDetector(model_config_path=str(tmp_path / "non_existent_model_pd.yaml"))
    
    model_path, _ = dummy_model_and_axiom_configs_path_pd(tmp_path_factory=pytest.TempPathFactory())
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        ParadoxDetector(model_config_path=model_path, axioms_config_path=str(tmp_path / "non_existent_axioms_pd.yaml"))
    logger.info("Test: FileNotFoundError correctly raised for missing config files.")
