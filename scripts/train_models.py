# ontological-playground-designer/scripts/train_models.py

import sys
import os
import typer
from typing import Optional, Dict, Any, List
import datetime
import json
import yaml # For loading YAML configs
import random
import numpy as np
import torch # For mock model saving/loading

# Ensure project_root is in sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'src'))

# Setup logging for the script
from src.utils.logger import setup_logging, logger
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_LOG_FILE = os.path.join(LOG_DIR, f"train_models_script_{TIMESTAMP}.log")
setup_logging(log_level="INFO", log_file=SCRIPT_LOG_FILE)


app = typer.Typer(
    name="Ontological Playground Training Script",
    help="Automates generation of training data and training/fine-tuning of AI models.",
    pretty_exceptions_enable=False # For detailed stack traces in logs
)

# --- Global AI Component Initialization ---
AXIOMS_CONFIG_PATH = "config/axioms.yaml"
MODEL_CONFIG_PATH = "config/model_config.yaml"
SIM_SETTINGS_PATH = "config/simulation_settings.yaml"

GENERATED_WORLDS_DIR = "data/generated_worlds"
EVALUATION_REPORTS_DIR = "data/evaluation_reports"
SIM_LOGS_DIR = "data/sim_logs"
MODELS_SAVE_DIR = "models" # Base directory for saving trained models

os.makedirs(GENERATED_WORLDS_DIR, exist_ok=True)
os.makedirs(EVALUATION_REPORTS_DIR, exist_ok=True)
os.makedirs(SIM_LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_SAVE_DIR, exist_ok=True)


try:
    from src.core.axiom_parser import AxiomParser, AxiomSet, ParsedAxiom
    from src.core.rule_generator import RuleGenerator, GeneratedWorldRules, GeneratedRule
    from src.core.world_compiler import WorldCompiler
    from src.core.flourishing_evaluator import FlourishingEvaluator, WorldEvaluationReport
    from src.core.paradox_detector import ParadoxDetector, ParadoxDetectionReport
    from simulators.simulator_adapter import SimulatorAdapter

    AXIOM_PARSER_INST = AxiomParser(model_config_path=MODEL_CONFIG_PATH)
    RULE_GENERATOR_INST = RuleGenerator(model_config_path=MODEL_CONFIG_PATH, simulation_settings_path=SIM_SETTINGS_PATH)
    WORLD_COMPILER_INST = WorldCompiler(simulation_settings_path=SIM_SETTINGS_PATH)
    FLOURISHING_EVALUATOR_INST = FlourishingEvaluator(model_config_path=MODEL_CONFIG_PATH, axioms_config_path=AXIOMS_CONFIG_PATH)
    PARADOX_DETECTOR_INST = ParadoxDetector(model_config_path=MODEL_CONFIG_PATH, axioms_config_path=AXIOMS_CONFIG_PATH)
    
    # Simulators for data generation (if needed to simulate worlds for ground truth)
    SIMULATOR_ADAPTER_INST = SimulatorAdapter(simulator_type="template_simulator", output_log_path=SIM_LOGS_DIR)
    
    logger.info("All core AI components initialized for training script.")
except Exception as e:
    logger.error(f"Failed to initialize one or more core AI components for script: {e}", exc_info=True)
    logger.error("Please ensure all configuration files are valid and models can be loaded. Exiting.")
    raise typer.Exit(code=1)

# --- Helper Functions ---
def _load_config_file(file_path: str) -> Dict[str, Any]:
    """Loads a JSON or YAML configuration file."""
    if not os.path.exists(file_path):
        logger.error(f"Config file not found: {file_path}")
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, 'r') as f:
            if file_extension == '.json':
                return json.load(f)
            elif file_extension == '.yaml' or file_extension == '.yml':
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_extension}. Use .json or .yaml.")
    except Exception as e:
        logger.error(f"Error loading config file '{file_path}': {e}", exc_info=True)
        raise

def _reconstruct_generated_rules_for_paradox_detector(compiled_config: Dict[str, Any]) -> GeneratedWorldRules:
    """
    Reconstructs a minimal GeneratedWorldRules object from compiled_config
    for use by the ParadoxDetector.
    """
    rules_list = []
    for category_list in compiled_config.get('generated_world_rules', {}).values():
        for r in category_list:
            rules_list.append(GeneratedRule(
                id=r['id'], description=r['description'], type=r['type'], 
                parameters=r.get('parameters', {}), dependencies=r.get('dependencies', []), 
                axiom_influence=r.get('axiom_influence', {})
            ))
    
    rule_graph_data = compiled_config.get('rule_interdependencies_graph')
    rule_graph = None
    if rule_graph_data:
        from src.utils.graph_utils import reconstruct_graph_from_json_serializable
        rule_graph = reconstruct_graph_from_json_serializable(rule_graph_data)

    return GeneratedWorldRules(
        world_name=compiled_config['world_metadata']['name'],
        rules=rules_list,
        rule_graph=rule_graph,
        creation_timestamp=compiled_config['world_metadata']['creation_timestamp'],
        axioms_used_ids=compiled_config['world_metadata']['axioms_influencing_design']
    )


def _generate_synthetic_training_data(num_samples: int, axiom_file: str) -> List[Dict[str, Any]]:
    """
    Generates synthetic training data by designing and evaluating worlds.
    This simulates the AI's "experience" for learning.
    Each sample is a dict containing compiled_config, axiom_set, paradox_report, evaluation_report.
    """
    logger.info(f"Generating {num_samples} synthetic training data samples...")
    training_data: List[Dict[str, Any]] = []

    axiom_set = AXIOM_PARSER_INST.parse_axioms(axiom_file)
    logger.debug("Axioms parsed for data generation.")

    for i in range(num_samples):
        world_seed_name = f"synth_world_{TIMESTAMP}_{i}"
        logger.debug(f"Generating data for synthetic world: {world_seed_name}")

        try:
            # 1. Generate Rules (RuleGenerator's mock logic gives diversity)
            generated_rules = RULE_GENERATOR_INST.generate_rules(axiom_set, world_seed_name)

            # 2. Compile World
            compiled_config = WORLD_COMPILER_INST.compile_world(generated_rules)
            
            # 3. Detect Paradoxes
            paradox_report = PARADOX_DETECTOR_INST.detect_paradoxes(generated_rules, axiom_set) # Use GeneratedWorldRules directly
            
            # 4. Evaluate Flourishing (Pass ground truth or simulated ground truth)
            # For evaluator training, we need 'ground truth' flourishing values.
            # In a real system, this would come from running simulations and observing outcomes.
            # Here, we'll let the FlourishingEvaluator's mock logic generate plausible GT based on rules.
            evaluation_report = FLOURISHING_EVALUATOR_INST.evaluate_world(
                compiled_config, axiom_set, paradox_risk_score=paradox_report.total_paradox_risk_score
            )
            
            training_data.append({
                "compiled_config": compiled_config,
                "axiom_set": [a.__dict__ for a in axiom_set.axioms], # Store axiom data
                "paradox_report": paradox_report.__dict__,
                "evaluation_report": evaluation_report.__dict__,
            })
            logger.debug(f"Generated synthetic data sample {i+1}/{num_samples}")

        except Exception as e:
            logger.error(f"Error generating synthetic data for '{world_seed_name}': {e}", exc_info=True)
            continue # Skip this sample if an error occurs

    logger.success(f"Successfully generated {len(training_data)} synthetic training data samples.")
    return training_data

# --- Mock Model Implementations for Training ---
# In a real scenario, these would be actual PyTorch/TensorFlow modules.
# Here, they simulate training by just printing messages and saving dummy data.

class MockGraphTransformerModel(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dummy_param = torch.nn.Parameter(torch.randn(10)) # A dummy trainable parameter
        logger.debug("MockGraphTransformerModel initialized.")

    def forward(self, axiom_embeddings: List[np.ndarray], target_rules_graph: Optional[Any] = None) -> Any:
        # Simulate rule generation or graph prediction
        return "Simulated rules graph output based on axioms."

    def train_step(self, data: Dict[str, Any]):
        # Simulate a training step
        loss = random.random() # Mock loss
        logger.debug(f"MockGraphTransformerModel training step. Loss: {loss:.4f}")
        return loss
    
    def save(self, path: str):
        logger.info(f"MockGraphTransformerModel saved to: {path}")
        torch.save(self.state_dict(), path) # Save dummy state_dict

class MockTimeDistributedGraphCNNModel(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dummy_param = torch.nn.Parameter(torch.randn(10))
        logger.debug("MockTimeDistributedGraphCNNModel initialized.")

    def forward(self, features: Dict[str, Any]) -> Dict[str, float]:
        # Simulate prediction of flourishing metrics
        return {
            'total_flourishing_score': random.uniform(0.5, 1.0),
            'sustainability_index': random.uniform(0.5, 1.0),
            'equity_distribution': random.uniform(0.5, 1.0),
            'paradox_risk_score_input': features.get('paradox_risk_score', 0.0) # Take input paradox score
        }

    def train_step(self, data: Dict[str, Any]):
        # Simulate a training step
        loss = random.random() # Mock loss
        logger.debug(f"MockTimeDistributedGraphCNNModel training step. Loss: {loss:.4f}")
        return loss

    def save(self, path: str):
        logger.info(f"MockTimeDistributedGraphCNNModel saved to: {path}")
        torch.save(self.state_dict(), path)

class MockGraphAttentionNetworkModel(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dummy_param = torch.nn.Parameter(torch.randn(10))
        logger.debug("MockGraphAttentionNetworkModel initialized.")

    def forward(self, rule_graph_data: Any) -> float:
        # Simulate paradox risk prediction
        return random.uniform(0.0, 0.5) # Mock risk score

    def train_step(self, data: Dict[str, Any]):
        # Simulate a training step
        loss = random.random() # Mock loss
        logger.debug(f"MockGraphAttentionNetworkModel training step. Loss: {loss:.4f}")
        return loss

    def save(self, path: str):
        logger.info(f"MockGraphAttentionNetworkModel saved to: {path}")
        torch.save(self.state_dict(), path)

# --- Training Orchestration ---

def _train_model(model: torch.nn.Module, data: List[Dict[str, Any]], model_name: str, epochs: int):
    """Generic mock training loop."""
    logger.info(f"Starting mock training for {model_name} for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for sample in data:
            # In a real scenario, format sample into model input tensors
            loss = model.train_step(sample)
            total_loss += loss
        avg_loss = total_loss / len(data) if data else 0
        logger.info(f"  {model_name} Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    logger.success(f"Mock training for {model_name} completed.")

# --- CLI Commands ---

@app.command(name="generate-data", help="[bold purple]Generate synthetic training data by designing and evaluating worlds.[/bold purple]")
def generate_training_data_command(
    num_samples: int = typer.Argument(10, help="[bold]Number of synthetic world designs to generate for training data.[/bold]"),
    axiom_file: str = typer.Option(AXIOMS_CONFIG_PATH, "--axioms", "-a", help="Path to the YAML file defining the axioms to use for data generation."),
    output_file: str = typer.Option(os.path.join("data", "training_data", f"synthetic_data_{TIMESTAMP}.jsonl"), "--output", "-o", help="Path to save the generated training data (JSONL).")
):
    """
    Generates a dataset of AI-designed worlds and their evaluations, suitable for training.
    """
    logger.info(f"--- Starting data generation for training ({num_samples} samples) ---")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    training_data = _generate_synthetic_training_data(num_samples, axiom_file)

    try:
        with open(output_file, 'w') as f:
            for sample in training_data:
                f.write(json.dumps(sample) + '\n')
        logger.success(f"[bold green]Synthetic training data saved to: {output_file}[/bold green]")
    except Exception as e:
        logger.error(f"Error saving training data: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command(name="train-all-models", help="[bold green]Train or fine-tune all core AI models.[/bold green]")
def train_all_models_command(
    data_file: str = typer.Argument(os.path.join("data", "training_data", f"synthetic_data_{TIMESTAMP}.jsonl"), help="[bold]Path to the JSONL file containing synthetic training data.[/bold]"),
    epochs_rule_gen: int = typer.Option(10, "--epochs-rule-gen", help="Number of epochs for RuleGenerator training."),
    epochs_flourish_eval: int = typer.Option(15, "--epochs-flourish-eval", help="Number of epochs for FlourishingEvaluator training."),
    epochs_paradox_det: int = typer.Option(12, "--epochs-paradox-det", help="Number of epochs for ParadoxDetector training.")
):
    """
    Orchestrates the training/fine-tuning of the RuleGenerator, FlourishingEvaluator,
    and ParadoxDetector models.
    """
    logger.info(f"--- Starting training pipeline using data from: '{data_file}' ---")
    
    if not os.path.exists(data_file):
        logger.error(f"Training data file not found: {data_file}. Please run 'generate-data' first.")
        raise typer.Exit(code=1)
    
    # Load training data
    training_data_loaded: List[Dict[str, Any]] = []
    try:
        with open(data_file, 'r') as f:
            for line in f:
                training_data_loaded.append(json.loads(line))
        logger.info(f"Loaded {len(training_data_loaded)} samples from '{data_file}'.")
    except Exception as e:
        logger.error(f"Error loading training data from '{data_file}': {e}", exc_info=True)
        raise typer.Exit(code=1)

    # Initialize mock models
    rule_gen_model = MockGraphTransformerModel(RULE_GENERATOR_INST.model_config['rule_generator_model'])
    flourish_eval_model = MockTimeDistributedGraphCNNModel(FLOURISHING_EVALUATOR_INST.model_config['flourishing_evaluator_model'])
    paradox_det_model = MockGraphAttentionNetworkModel(PARADOX_DETECTOR_INST.model_config['paradox_detector_model'])

    # --- 1. Train RuleGenerator Model ---
    logger.info("\n--- Training RuleGenerator Model ---")
    # RuleGenerator's training data would involve axiom embeddings + target rule graphs
    # This is a simplification; in reality, data would be structured differently
    rule_gen_training_data = []
    for sample in training_data_loaded:
        # Mock: RuleGenerator learns to generate rule graphs given axiom_set
        # We simulate this by simply passing axioms and the generated rules graph from the sample
        axiom_set_for_rule_gen = AxiomSet(axioms=[ParsedAxiom(**a_data) for a_data in sample['axiom_set']])
        rule_gen_training_data.append({
            "axiom_set": axiom_set_for_rule_gen,
            "target_rules_graph": _reconstruct_generated_rules_for_paradox_detector(sample['compiled_config']).rule_graph
        })
    _train_model(rule_gen_model, rule_gen_training_data, "RuleGenerator", epochs_rule_gen)
    rule_gen_model.save(os.path.join(MODELS_SAVE_DIR, "rule_generator_model.pth"))

    # --- 2. Train FlourishingEvaluator Model ---
    logger.info("\n--- Training FlourishingEvaluator Model ---")
    # FlourishingEvaluator's training data would involve features from compiled_config + ground truth evaluation_report
    flourish_eval_training_data = []
    for sample in training_data_loaded:
        features = FLOURISHING_EVALUATOR_INST._extract_features_from_world_config(sample['compiled_config'])
        features['paradox_risk_score'] = sample['paradox_report']['total_paradox_risk_score'] # Include this
        flourish_eval_training_data.append({
            "features": features,
            "ground_truth_metrics": sample['evaluation_report']['predicted_metrics'] # Use this as GT
        })
    _train_model(flourish_eval_model, flourish_eval_training_data, "FlourishingEvaluator", epochs_flourish_eval)
    flourish_eval_model.save(os.path.join(MODELS_SAVE_DIR, "flourishing_evaluator_model.pth"))

    # --- 3. Train ParadoxDetector Model ---
    logger.info("\n--- Training ParadoxDetector Model ---")
    # ParadoxDetector's training data would involve rule graphs + ground truth paradox labels
    paradox_det_training_data = []
    for sample in training_data_loaded:
        paradox_det_training_data.append({
            "world_rules": _reconstruct_generated_rules_for_paradox_detector(sample['compiled_config']),
            "ground_truth_paradoxes": sample['paradox_report']['detected_paradoxes']
        })
    _train_model(paradox_det_model, paradox_det_training_data, "ParadoxDetector", epochs_paradox_det)
    paradox_det_model.save(os.path.join(MODELS_SAVE_DIR, "paradox_detector_model.pth"))

    logger.success("[bold green]All models trained/fine-tuned and saved successfully.[/bold green]")


def main():
    app()

if __name__ == "__main__":
    main()
