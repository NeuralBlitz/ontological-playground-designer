# ontological-playground-designer/simulators/simulator_adapter.py

import json
import yaml
import os
from typing import Dict, Any, List, Type, Optional

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Import the TemplateSimulator for integration example
from simulators.template_simulator.template_simulator import TemplateSimulator

# Setup logging for this module
setup_logging()

# --- Type Hinting for generic simulator classes ---
# Define a protocol or ABC if multiple simulator types are expected
# For simplicity, we'll assume simulators have a similar `__init__` and `run_simulation` method.
# A more robust solution might use an abstract base class.
GenericSimulatorClass = Any # Placeholder for a class that can be instantiated
GenericSimulatorInstance = Any # Placeholder for an instantiated simulator object

class SimulatorAdapter:
    """
    Provides a standardized interface for running AI-designed world configurations
    in various simulation engines.

    This adapter abstracts away simulator-specific details, allowing the
    Ontological Playground Designer to remain flexible and integrate with
    different simulation technologies. It's the "manifestation middleware"
    that ensures our ontological blueprints can execute.
    """
    def __init__(self, simulator_type: str = "template_simulator",
                 output_log_path: str = "data/sim_logs"):
        """
        Initializes the SimulatorAdapter for a specific simulator type.

        Args:
            simulator_type (str): The identifier for the simulation engine to use
                                  (e.g., "template_simulator", "netlogo_adapter", "unity_adapter").
            output_log_path (str): Base directory where simulation logs should be saved.
        """
        self.simulator_type = simulator_type
        self.output_log_path = output_log_path
        self._simulator_class: Optional[GenericSimulatorClass] = None
        self._load_simulator_class()
        logger.info(f"SimulatorAdapter initialized for type: '{self.simulator_type}'")

    def _load_simulator_class(self):
        """
        Dynamically loads the appropriate simulator class based on `self.simulator_type`.
        This allows for easy extension to other simulators.
        """
        if self.simulator_type == "template_simulator":
            self._simulator_class = TemplateSimulator
            logger.debug(f"Loaded TemplateSimulator for type '{self.simulator_type}'.")
        # --- Future Extensions ---
        # elif self.simulator_type == "netlogo_adapter":
        #     from simulators.netlogo_adapter.netlogo_adapter import NetLogoSimulator
        #     self._simulator_class = NetLogoSimulator
        #     logger.debug(f"Loaded NetLogoSimulator for type '{self.simulator_type}'.")
        # elif self.simulator_type == "unity_adapter":
        #     from simulators.unity_adapter.unity_adapter import UnitySimulator
        #     self._simulator_class = UnitySimulator
        #     logger.debug(f"Loaded UnitySimulator for type '{self.simulator_type}'.")
        else:
            logger.error(f"Unsupported simulator type: '{self.simulator_type}'.")
            raise ValueError(f"Unsupported simulator type: '{self.simulator_type}'.")

    def run_world(self, compiled_world_config: Dict[str, Any], log_interval: int = 100) -> str:
        """
        Instantiates and runs a simulation using the loaded simulator class
        and the provided world configuration.

        Args:
            compiled_world_config (Dict[str, Any]): The complete simulation configuration
                                                    (output from WorldCompiler).
            log_interval (int): How often (in steps) to log the full simulation state.

        Returns:
            str: The path to the generated simulation log file.
        """
        if not self._simulator_class:
            logger.error("No simulator class loaded. Cannot run world.")
            raise RuntimeError("Simulator adapter not properly initialized.")

        world_name = compiled_world_config.get('world_metadata', {}).get('name', 'UnnamedWorld')
        logger.info(f"Running world '{world_name}' using simulator type: '{self.simulator_type}'...")

        try:
            # Instantiate the specific simulator with the world configuration
            simulator_instance: GenericSimulatorInstance = self._simulator_class(world_config=compiled_world_config)
            
            # Run the simulation
            log_file_path = simulator_instance.run_simulation(
                output_log_path=self.output_log_path,
                log_interval=log_interval
            )
            logger.success(f"Simulation for '{world_name}' completed successfully. Log: {log_file_path}")
            return log_file_path
        except Exception as e:
            logger.error(f"Error running simulation for world '{world_name}' with {self.simulator_type}: {e}", exc_info=True)
            raise

# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    import os
    # Ensure src/utils directory and logger.py exist for setup_logging
    # Also ensure simulators/template_simulator directory and template_simulator.py exist
    if not os.path.exists("src/utils"): os.makedirs("src/utils")
    if not os.path.exists("simulators/template_simulator"): os.makedirs("simulators/template_simulator")
    if not os.path.exists("data/sim_logs"): os.makedirs("data/sim_logs")
    
    # Create a dummy template_simulator.py if it's not there (to allow import)
    template_sim_path = "simulators/template_simulator/template_simulator.py"
    if not os.path.exists(template_sim_path):
        with open(template_sim_path, 'w') as f:
            f.write("""
# simulators/template_simulator/template_simulator.py (DUMMY for adapter test)
import json
import random
from typing import Dict, Any, List
import datetime
from loguru import logger
from dataclasses import dataclass, field

@dataclass
class AgentState: id: str; species: str; x: int; y: int; energy: float; well_being: float; cooperation_tendency: float; is_alive: bool = True
@dataclass
class ResourceNode: x: int; y: int; type: str; amount: float; regen_rate: float
@dataclass
class SimulationState: time_step: int; agents: List[AgentState]; resources: List[ResourceNode]; world_metrics: Dict[str, Any]; log_messages: List[str] = field(default_factory=list)

class TemplateSimulator:
    def __init__(self, world_config: Dict[str, Any]):
        self.world_config = world_config
        self.world_name = world_config.get('world_metadata', {}).get('name', 'UnnamedWorld')
        self.sim_settings = world_config.get('simulation_defaults', {})
        self.max_simulation_steps = self.sim_settings.get('max_simulation_steps', 100)
        self.agents = [AgentState(id=f"A{i}", species="S1", x=0, y=0, energy=100, well_being=0.5, cooperation_tendency=0.5) for i in range(5)]
        self.resources = [ResourceNode(x=0, y=0, type="R1", amount=100, regen_rate=0.01)]
        logger.info(f"Dummy TemplateSimulator for '{self.world_name}' initialized.")

    def _serialize_state(self, state: SimulationState) -> Dict[str, Any]:
        return {k: v if not isinstance(v, list) else [item.__dict__ for item in v] for k, v in state.__dict__.items()}

    def run_simulation(self, output_log_path: str = "data/sim_logs", log_interval: int = 10):
        log_filename = os.path.join(output_log_path, f"{self.world_name}_sim_log_DUMMY_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        current_state = SimulationState(time_step=0, agents=self.agents, resources=self.resources, world_metrics={'live_agent_count': len(self.agents)})
        
        with open(log_filename, 'w') as f_log:
            f_log.write(json.dumps(self._serialize_state(current_state)) + '\\n')
            for step in range(1, self.max_simulation_steps + 1):
                if random.random() < 0.1: # Simulate some agent death
                    if current_state.agents and current_state.agents[0].is_alive:
                        current_state.agents[0].is_alive = False
                        current_state.world_metrics['live_agent_count'] -= 1
                if step % log_interval == 0 or step == self.max_simulation_steps:
                    f_log.write(json.dumps(self._serialize_state(current_state)) + '\\n')
                if current_state.world_metrics['live_agent_count'] <= 0: break # All dead
        logger.success(f"Dummy simulation for '{self.world_name}' completed. Log saved to: {log_filename}")
        return log_filename
""")
        logger.info("Created dummy simulators/template_simulator/template_simulator.py for testing.")


    logger.info("--- Demonstrating SimulatorAdapter ---")

    # A minimal dummy compiled_world_config (as output by world_compiler.py)
    dummy_compiled_world_config = {
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
        'generated_world_rules': {} # Can be empty for this dummy test
    }

    # 1. Initialize adapter for the template simulator
    adapter = SimulatorAdapter(simulator_type="template_simulator")

    # 2. Run the world through the adapter
    try:
        sim_log_file = adapter.run_world(dummy_compiled_world_config, log_interval=10)
        logger.info(f"\n[bold green]World run via adapter. Simulation log: {sim_log_file}[/bold green]")
    except Exception as e:
        logger.error(f"Failed to run world via adapter: {e}")
    
    # --- Demonstrate with an unsupported simulator type ---
    logger.info("\n--- Demonstrating Unsupported Simulator Type ---")
    try:
        unsupported_adapter = SimulatorAdapter(simulator_type="non_existent_simulator")
        unsupported_adapter.run_world(dummy_compiled_world_config)
    except ValueError as e:
        logger.warning(f"Caught expected error for unsupported simulator: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for unsupported simulator: {e}", exc_info=True)
