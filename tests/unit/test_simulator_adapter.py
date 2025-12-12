# ontological-playground-designer/tests/unit/test_simulator_adapter.py

import pytest
import os
import json
import datetime
from unittest.mock import MagicMock, patch

# Ensure project_root is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulators.simulator_adapter import SimulatorAdapter
from src.utils.logger import setup_logging

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_simulator_adapter.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_sim_logs_dir(tmp_path_factory):
    """Creates a temporary directory for simulation logs."""
    log_dir = tmp_path_factory.mktemp("sim_logs")
    return str(log_dir)

@pytest.fixture
def mock_template_simulator_class():
    """Mocks the TemplateSimulator class for dynamic loading."""
    mock_simulator_instance = MagicMock()
    mock_simulator_instance.run_simulation.return_value = "mock/path/to/sim_log.jsonl"
    
    mock_simulator_class = MagicMock(return_value=mock_simulator_instance)
    yield mock_simulator_class

@pytest.fixture
def adapter_instance(mock_template_simulator_class, dummy_sim_logs_dir):
    """Returns a SimulatorAdapter instance with mocked TemplateSimulator."""
    with patch('simulators.simulator_adapter.TemplateSimulator', new=mock_template_simulator_class):
        adapter = SimulatorAdapter(simulator_type="template_simulator", output_log_path=dummy_sim_logs_dir)
        yield adapter

@pytest.fixture
def mock_compiled_world_config():
    """Returns a mock compiled world configuration dictionary."""
    return {
        'simulation_defaults': {
            'initial_world_size': {'x_dim': 50, 'y_dim': 50},
            'max_simulation_steps': 100,
            'initial_agent_count': 5,
            'simulation_engine_version': "generic_agent_based_v1.0"
        },
        'world_metadata': {
            'name': "AdapterTestWorld",
            'creation_timestamp': datetime.datetime.now().isoformat(),
            'axioms_influencing_design': ['DUMMY_AXIOM_001'],
            'designed_by': "SimulatorAdapter Test",
        },
        'generated_world_rules': {}
    }

# --- Test Cases ---

def test_simulator_adapter_initialization_valid_type(adapter_instance, mock_template_simulator_class):
    """Tests if the SimulatorAdapter initializes correctly with a valid type."""
    assert adapter_instance is not None
    assert adapter_instance.simulator_type == "template_simulator"
    assert adapter_instance._simulator_class == mock_template_simulator_class
    logger.info("Test: SimulatorAdapter initializes correctly with valid type.")

def test_simulator_adapter_initialization_unsupported_type(dummy_sim_logs_dir):
    """Tests if initialization fails for an unsupported simulator type."""
    with pytest.raises(ValueError, match="Unsupported simulator type"):
        SimulatorAdapter(simulator_type="non_existent_simulator", output_log_path=dummy_sim_logs_dir)
    logger.info("Test: SimulatorAdapter correctly raises error for unsupported type.")

def test_run_world_executes_simulator(adapter_instance, mock_compiled_world_config, mock_template_simulator_class):
    """Tests if run_world correctly instantiates and calls the simulator's run_simulation."""
    log_file = adapter_instance.run_world(mock_compiled_world_config, log_interval=50)
    
    # Verify the mock simulator class was instantiated
    mock_template_simulator_class.assert_called_once_with(world_config=mock_compiled_world_config)
    
    # Verify the run_simulation method on the mock instance was called
    mock_template_simulator_class.return_value.run_simulation.assert_called_once_with(
        output_log_path=adapter_instance.output_log_path,
        log_interval=50
    )
    assert log_file == "mock/path/to/sim_log.jsonl"
    logger.info("Test: run_world correctly executes the simulator.")

def test_run_world_no_simulator_loaded(dummy_sim_logs_dir):
    """Tests error handling if no simulator class was loaded."""
    adapter = SimulatorAdapter(simulator_type="template_simulator", output_log_path=dummy_sim_logs_dir)
    adapter._simulator_class = None # Manually unset to simulate failure
    with pytest.raises(RuntimeError, match="Simulator adapter not properly initialized."):
        adapter.run_world(mock_compiled_world_config)
    logger.info("Test: run_world correctly raises error if no simulator is loaded.")
