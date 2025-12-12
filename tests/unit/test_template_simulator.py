# ontological-playground-designer/tests/unit/test_template_simulator.py

import pytest
import os
import json
import datetime
import random
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure project_root is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulators.template_simulator.template_simulator import TemplateSimulator, AgentState, ResourceNode, SimulationState
from src.utils.logger import setup_logging

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_template_simulator.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_world_config():
    """Returns a mock compiled_world_config for the simulator."""
    return {
        'simulation_defaults': {
            'simulation_engine_version': "generic_agent_based_v1.0",
            'initial_world_size': {'x_dim': 10, 'y_dim': 10, 'z_dim': 1},
            'time_step_duration_ms': 100,
            'max_simulation_steps': 5, # Short for unit tests
            'initial_agent_count': 2,
            'agent_species_diversity': 1,
            'base_agent_energy_consumption': 1.0,
            'max_agent_lifespan_steps': 100,
            'base_resource_regen_rate': 0.1,
            'initial_agent_cooperation_tendency': 0.5,
            'environmental_diversity_index': 0.7,
            'environmental_axioms_active': True,
            'agent_axioms_active': True
        },
        'world_metadata': {
            'name': "TestSimWorld",
            'creation_timestamp': datetime.datetime.now().isoformat(),
            'axioms_influencing_design': ['DUMMY_AXIOM_001'],
            'designed_by': "TemplateSimulator Test",
        },
        'generated_world_rules': {
            'agent_behaviors': [
                {'id': 'Agent_Cooperation_Rule', 'description': 'Agents gain well-being from cooperative actions.', 'type': 'agent_behavior', 'parameters': {'cooperation_reward_factor': 1.5}}, # Boost cooperation
                {'id': 'Agent_HighEnergyLoss', 'description': 'Agents lose more energy than usual.', 'type': 'agent_behavior', 'parameters': {'energy_loss_multiplier': 2.0}}
            ],
            'environmental_laws': [
                {'id': 'Resource_Regen_Rule', 'description': 'Resource regeneration is boosted.', 'type': 'environmental_law', 'parameters': {'regen_health_multiplier': 2.0}} # Boost regen
            ],
            'system_mechanics': [
                {'id': 'Flourishing_Feedback', 'description': 'Boost resources if flourishing is low.', 'type': 'system_mechanic', 'parameters': {'resource_adjustment_sensitivity': 0.05}}
            ]
        }
    }

@pytest.fixture
def simulator_instance(dummy_world_config):
    """Returns a TemplateSimulator instance."""
    # Patch random.randint and random.uniform to make initialization deterministic for tests
    with patch('random.randint', side_effect=[0,0,0,0,0,0, # Agent pos, resource pos
                                             1 # species diversity
                                             ]), \
         patch('random.uniform', side_effect=[100.0, # Agent energy
                                              0.7, # Agent well-being
                                              50.0, 150.0, 0.1 # Resource amount/regen for 2 nodes
                                              ]):
        sim = TemplateSimulator(world_config=dummy_world_config)
        yield sim

@pytest.fixture
def dummy_sim_logs_dir(tmp_path_factory):
    """Creates a temporary directory for simulation logs."""
    log_dir = tmp_path_factory.mktemp("sim_logs_ts")
    return str(log_dir)

# --- Test Cases ---

def test_simulator_initialization(simulator_instance, dummy_world_config):
    """Tests if the simulator initializes its state correctly."""
    assert simulator_instance.world_name == "TestSimWorld"
    assert simulator_instance.current_time_step == 0
    assert simulator_instance.max_simulation_steps == 5
    assert len(simulator_instance.agents) == 2 # From config
    assert len(simulator_instance.resources) > 0 # Based on world size
    
    # Check if a rule influenced agent state during init
    assert simulator_instance.agents[0].cooperation_tendency == pytest.approx(0.75) # 0.5 * 1.5 from rule
    assert simulator_instance.resources[0].regen_rate == pytest.approx(0.02) # 0.01 * 2.0 from rule
    logger.info("Test: Simulator initializes world state correctly.")

def test_apply_rules_and_step_agent_movement_and_energy(simulator_instance):
    """Tests basic agent movement and energy consumption."""
    initial_agent = simulator_instance.agents[0]
    initial_energy = initial_agent.energy
    
    # Mock random choices for deterministic movement and resource interaction
    with patch('random.choice', side_effect=[0, 0, # Agent X,Y move
                                            simulator_instance.resources[0], # Consume this resource
                                            simulator_instance.agents[1], # Cooperate with this agent
                                            ]):
        # Ensure agent doesn't die immediately in mock
        initial_agent.energy = 200 # Give it plenty of energy
        # Patch energy loss multiplier from generated rules for specific test
        for rule_entry in simulator_instance.generated_rules.get('agent_behaviors', []):
            if 'Agent_HighEnergyLoss' in rule_entry['id']:
                rule_entry['parameters']['energy_loss_multiplier'] = 1.0 # Disable high loss for this specific test

        next_state = simulator_instance._apply_rules_and_step(simulator_instance.simulation_log[0])
        
        # Agent should have moved (mocked to 0,0 again if initial was 0,0)
        # Assuming initial was 0,0 and mock choice is 0,0 -> pos should be same
        # if actual pos changed (x,y in range 0-9), this confirms movement logic
        assert 0 <= next_state.agents[0].x < simulator_instance.world_size['x_dim']
        assert next_state.agents[0].y == initial_agent.y # Due to mock.choice above
        
        # Energy should have decreased due to base_agent_energy_consumption
        expected_energy_loss = simulator_instance.sim_settings['base_agent_energy_consumption']
        assert next_state.agents[0].energy < initial_energy - expected_energy_loss + 0.1 # Allow for resource gain
        
        logger.info("Test: Agent movement and base energy consumption in one step.")

def test_apply_rules_and_step_resource_regen(simulator_instance):
    """Tests resource regeneration logic."""
    initial_resource_amount = simulator_instance.resources[0].amount
    initial_regen_rate = simulator_instance.resources[0].regen_rate # Should be 0.02 due to rule
    
    # Need a dummy state to pass to _apply_rules_and_step
    dummy_state = SimulationState(
        time_step=0,
        agents=simulator_instance.agents,
        resources=[ResourceNode(x=r.x, y=r.y, type=r.type, amount=r.amount, regen_rate=r.regen_rate) for r in simulator_instance.resources],
        world_metrics={}
    )
    
    next_state = simulator_instance._apply_rules_and_step(dummy_state)
    
    # Resource amount should increase due to regen_rate
    # next_amount = initial_amount + initial_regen_rate * initial_amount
    expected_amount = initial_resource_amount + initial_regen_rate * initial_resource_amount
    assert next_state.resources[0].amount > initial_resource_amount
    assert next_state.resources[0].amount == pytest.approx(expected_amount)
    logger.info("Test: Resource regeneration in one step.")

def test_apply_rules_and_step_flourishing_feedback_rule(simulator_instance):
    """Tests if 'Flourishing_Feedback' system rule adjusts resources."""
    # Set up a state where avg_flourishing is low
    low_flourish_state = SimulationState(
        time_step=0,
        agents=[AgentState(id="A1", species="S1", x=0, y=0, energy=100, well_being=0.2, cooperation_tendency=0.5)],
        resources=[ResourceNode(x=0, y=0, type="BasicResource", amount=100, regen_rate=0.01)],
        world_metrics={'avg_agent_flourishing': 0.2} # Manually set low flourishing
    )
    initial_regen_rate = low_flourish_state.resources[0].regen_rate
    
    next_state = simulator_instance._apply_rules_and_step(low_flourish_state)
    
    # Regen rate should be boosted by the Flourishing_Feedback rule
    assert next_state.resources[0].regen_rate > initial_regen_rate
    logger.info("Test: Flourishing_Feedback rule correctly boosts resource regen.")

def test_agent_death(simulator_instance):
    """Tests if an agent dies when energy hits zero or below."""
    agent_to_kill = simulator_instance.agents[0]
    agent_to_kill.energy = 0.5 # Set to low energy
    
    # Mock random choices to ensure agent moves but doesn't consume enough
    with patch('random.choice', side_effect=[0, 0, # Agent X,Y move
                                            None, # No nearby resource to consume
                                            ]):
        # Also mock base_agent_energy_consumption to make sure it kills it
        simulator_instance.sim_settings['base_agent_energy_consumption'] = 1.0 # Ensure it's enough to kill
        
        current_state = SimulationState(
            time_step=0,
            agents=[agent_to_kill],
            resources=simulator_instance.resources,
            world_metrics={}
        )
        
        next_state = simulator_instance._apply_rules_and_step(current_state)
        
        assert not next_state.agents[0].is_alive
        assert next_state.agents[0].well_being == 0.0
        assert "died" in next_state.log_messages[0]
        logger.info("Test: Agent correctly dies when energy runs out.")

def test_run_simulation_logs_correctly(simulator_instance, dummy_sim_logs_dir):
    """Tests if run_simulation generates a log file with correct format and content."""
    log_file = simulator_instance.run_simulation(output_log_path=dummy_sim_logs_dir, log_interval=1)
    
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == simulator_instance.max_simulation_steps + 1 # Initial + 5 steps
    
    first_state = json.loads(lines[0])
    assert first_state['time_step'] == 0
    assert 'agents' in first_state
    
    last_state = json.loads(lines[-1])
    assert last_state['time_step'] == simulator_instance.max_simulation_steps
    logger.info("Test: run_simulation logs correctly and saves to file.")

def test_run_simulation_stops_on_no_live_agents(simulator_instance, dummy_sim_logs_dir):
    """Tests if simulation stops when all agents die."""
    simulator_instance.sim_settings['max_simulation_steps'] = 100 # Long duration
    simulator_instance.agents[0].energy = 0.1 # Make them die quickly
    simulator_instance.agents[1].energy = 0.1
    simulator_instance.sim_settings['base_agent_energy_consumption'] = 100 # Guaranteed death

    log_file = simulator_instance.run_simulation(output_log_path=dummy_sim_logs_dir, log_interval=1)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    last_logged_state = json.loads(lines[-1])
    assert last_logged_state['world_metrics']['live_agent_count'] == 0
    assert len(lines) < simulator_instance.max_simulation_steps + 1 # Should stop early
    logger.info("Test: Simulation stops when all agents die.")

def test_serialize_state(simulator_instance):
    """Tests if SimulationState can be correctly serialized to JSON."""
    state = SimulationState(
        time_step=1,
        agents=[AgentState(id="A1", species="S1", x=1, y=1, energy=50, well_being=0.6, cooperation_tendency=0.5, is_alive=True)],
        resources=[ResourceNode(x=5, y=5, type="R1", amount=100, regen_rate=0.01)],
        world_metrics={'test_metric': 123},
        log_messages=["Agent A1 moved"]
    )
    serialized = simulator_instance._serialize_state(state)
    
    assert isinstance(serialized, dict)
    assert serialized['time_step'] == 1
    assert serialized['world_metrics']['test_metric'] == 123
    assert isinstance(serialized['agents'][0], dict) # AgentState should be converted to dict
    assert serialized['agents'][0]['id'] == "A1"
    logger.info("Test: _serialize_state correctly converts SimulationState to dict.")
