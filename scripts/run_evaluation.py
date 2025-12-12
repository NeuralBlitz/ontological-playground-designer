# ontological-playground-designer/scripts/run_evaluation.py

import sys
import os
import typer
from typing import Optional, Dict, Any
import datetime
import json
import yaml # For loading YAML configs

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
SCRIPT_LOG_FILE = os.path.join(LOG_DIR, f"run_evaluation_script_{TIMESTAMP}.log")
setup_logging(log_level="INFO", log_file=SCRIPT_LOG_FILE)


app = typer.Typer(
    name="Ontological Playground Evaluation Script",
    help="Automates the evaluation of an AI-designed world, including optional simulation.",
    pretty_exceptions_enable=False # For detailed stack traces in logs
)

# --- Global AI Component Initialization ---
# These components are initialized once at script start, mirroring CLI behavior.
AXIOMS_CONFIG_PATH = "config/axioms.yaml"
MODEL_CONFIG_PATH = "config/model_config.yaml"
SIM_SETTINGS_PATH = "config/simulation_settings.yaml"

GENERATED_WORLDS_DIR = "data/generated_worlds"
EVALUATION_REPORTS_DIR = "data/evaluation_reports"
SIM_LOGS_DIR = "data/sim_logs"

os.makedirs(EVALUATION_REPORTS_DIR, exist_ok=True)
os.makedirs(SIM_LOGS_DIR, exist_ok=True)


try:
    from src.core.axiom_parser import AxiomParser, AxiomSet
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
    
    # Default simulator adapter (can be overridden)
    SIMULATOR_ADAPTER_INST = SimulatorAdapter(simulator_type="template_simulator", output_log_path=SIM_LOGS_DIR)
    
    logger.info("All core AI components initialized for evaluation script.")
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
    
    # Rule graph is not strictly needed for the mock paradox detector logic if not provided directly
    rule_graph_data = compiled_config.get('rule_interdependencies_graph')
    rule_graph = None
    if rule_graph_data:
        # Minimal graph reconstruction if needed by actual model
        from src.utils.graph_utils import reconstruct_graph_from_json_serializable
        rule_graph = reconstruct_graph_from_json_serializable(rule_graph_data)

    return GeneratedWorldRules(
        world_name=compiled_config['world_metadata']['name'],
        rules=rules_list,
        rule_graph=rule_graph,
        creation_timestamp=compiled_config['world_metadata']['creation_timestamp'],
        axioms_used_ids=compiled_config['world_metadata']['axioms_influencing_design']
    )

# --- CLI Commands ---

@app.command(name="evaluate-world", help="[bold blue]Run a full evaluation pipeline for a compiled world.[/bold blue]")
def evaluate_world_command(
    world_config_path: str = typer.Argument(..., help="[bold]Path to the compiled world configuration file (JSON/YAML).[/bold]"),
    axiom_file: str = typer.Option(AXIOMS_CONFIG_PATH, "--axioms", "-a", help="Path to the YAML file defining the axioms used."),
    run_simulation: bool = typer.Option(False, "--simulate", "-s", help="Run a simulation before evaluation if no log exists."),
    simulation_log_path: Optional[str] = typer.Option(None, "--sim-log", help="[Optional] Path to an existing simulation log file (JSONL)."),
    simulator_type: str = typer.Option("template_simulator", "--simulator", help="Type of simulator to use if --simulate is active."),
    output_report_dir: str = typer.Option(EVALUATION_REPORTS_DIR, "--report-dir", "-o", help="Directory to save the evaluation report.")
):
    """
    Automates the evaluation of an AI-designed world.
    Optionally runs a simulation if a log is not provided or --simulate is active.
    """
    logger.info(f"--- Starting Evaluation Pipeline for: '{world_config_path}' ---")
    
    try:
        # 1. Load Compiled World Configuration
        compiled_config = _load_config_file(world_config_path)
        world_name = compiled_config['world_metadata']['name']
        logger.info(f"Loaded compiled world configuration for '{world_name}'.")

        # 2. Handle Simulation (if required)
        final_sim_log_path = simulation_log_path
        if run_simulation or (simulation_log_path and not os.path.exists(simulation_log_path)):
            logger.info("Simulation requested or log not found. Running simulation...")
            sim_adapter = SimulatorAdapter(simulator_type=simulator_type, output_log_path=SIM_LOGS_DIR)
            final_sim_log_path = sim_adapter.run_world(compiled_config)
            logger.success(f"Simulation completed. Log saved to: {final_sim_log_path}")
        elif not simulation_log_path and not run_simulation:
            logger.warning("No simulation log provided and --simulate not active. Evaluation will proceed without simulation dynamics.")
            # For this scenario, evaluators would make predictions based purely on static config
            
        # 3. Parse Axioms
        axiom_set = AXIOM_PARSER_INST.parse_axioms(axiom_file)
        logger.info("Axioms parsed.")

        # 4. Detect Paradoxes (using the generated rules from compiled_config)
        generated_rules_for_paradox = _reconstruct_generated_rules_for_paradox_detector(compiled_config)
        paradox_report = PARADOX_DETECTOR_INST.detect_paradoxes(generated_rules_for_paradox, axiom_set)
        logger.info(f"Paradox detection completed. Total risk: {paradox_report.total_paradox_risk_score:.2f}")
        
        # 5. Evaluate Flourishing
        # Note: For full simulation dynamics in evaluation, the evaluator would need to load
        # and process the `final_sim_log_path` which is a future enhancement.
        # For now, FLOURISHING_EVALUATOR_INST.evaluate_world only takes compiled_config
        # and paradox_risk_score for its mock prediction.
        evaluation_report = FLOURISHING_EVALUATOR_INST.evaluate_world(
            compiled_config, axiom_set, paradox_risk_score=paradox_report.total_paradox_risk_score
        )
        logger.info(f"Flourishing evaluation completed. Overall score: {evaluation_report.overall_flourishing_score:.2f}")

        # 6. Save Evaluation Report
        report_filename = os.path.join(output_report_dir, f"{world_name}_full_report_{TIMESTAMP}.txt")
        with open(report_filename, 'w') as f:
            f.write(f"--- World Evaluation Report for: {world_name} ---\n")
            f.write(f"Evaluation Timestamp: {evaluation_report.evaluation_timestamp}\n")
            f.write(f"Overall Flourishing Score: {evaluation_report.overall_flourishing_score:.2f}\n")
            f.write(f"Paradox Risk Score: {evaluation_report.paradox_risk_score:.2f}\n")
            f.write(f"Simulation Log Used: {final_sim_log_path if final_sim_log_path else 'None (static prediction)'}\n")
            f.write("\n--- Axiom Adherence Scores ---\n")
            for axiom_id, score in evaluation_report.axiom_adherence_scores.items():
                f.write(f"- {axiom_id}: {score:.2f}\n")
            f.write("\n--- Predicted Metrics ---\n")
            for metric in evaluation_report.predicted_metrics:
                f.write(f"- {metric.name}: {metric.value:.2f} ({metric.interpretation})\n")
            f.write("\n--- Detected Paradoxes ---\n")
            if paradox_report.detected_paradoxes:
                for paradox in paradox_report.detected_paradoxes:
                    f.write(f"- ID: {paradox.id}, Type: {paradox.type}, Severity: {paradox.severity:.2f}\n")
                    f.write(f"  Desc: {paradox.description}\n")
                    f.write(f"  Involved Rules: {paradox.involved_rules_ids}\n")
                    if paradox.conflict_path_description:
                        f.write(f"  Path: {paradox.conflict_path_description}\n")
                    if paradox.suggested_resolution:
                        f.write(f"  Suggestion: {paradox.suggested_resolution}\n")
            else:
                f.write("No major paradoxes detected.\n")
            f.write("\n--- Recommendations ---\n")
            for rec in evaluation_report.recommendations:
                f.write(rec + "\n")
        
        logger.success(f"[bold green]Full evaluation pipeline for '{world_name}' completed. Report: {report_filename}[/bold green]")
    
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        typer.echo(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"Error: {e}")
        typer.echo(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation pipeline: {e}", exc_info=True)
        typer.echo(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)


def main():
    app()

if __name__ == "__main__":
    main()
