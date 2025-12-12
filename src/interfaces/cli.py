# ontological-playground-designer/src/interfaces/cli.py

import typer
from typing import Optional, List
import os
import datetime
import json
import yaml # For saving/loading simulation settings
import shutil # For creating log directories

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Import core AI components
from src.core.axiom_parser import AxiomParser, AxiomSet
from src.core.rule_generator import RuleGenerator, GeneratedWorldRules
from src.core.world_compiler import WorldCompiler
from src.core.flourishing_evaluator import FlourishingEvaluator, WorldEvaluationReport
from src.core.paradox_detector import ParadoxDetector, ParadoxDetectionReport

# Setup logging for the CLI, ensuring it's available early
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
GLOBAL_LOG_FILE = os.path.join(LOG_DIR, f"opd_session_{TIMESTAMP}.log")
setup_logging(log_level="INFO", log_file=GLOBAL_LOG_FILE)


app = typer.Typer(
    name="Ontological Playground Designer",
    help="AI for axiomatically aligned simulation genesis. Design worlds from their foundational rules up.",
    rich_markup_enable=True
)

# --- Global AI Component Initialization ---
# These components are initialized once when the CLI app starts.
# Their configurations are loaded from the 'config' directory.
# This mimics the persistent, self-maintaining nature of the Omega Prime Reality.

# Configuration paths (relative to project root)
AXIOMS_CONFIG_PATH = "config/axioms.yaml"
MODEL_CONFIG_PATH = "config/model_config.yaml"
SIM_SETTINGS_PATH = "config/simulation_settings.yaml"

# Output paths
GENERATED_WORLDS_DIR = "data/generated_worlds"
EVALUATION_REPORTS_DIR = "data/evaluation_reports"
OS.makedirs(GENERATED_WORLDS_DIR, exist_ok=True)
OS.makedirs(EVALUATION_REPORTS_DIR, exist_ok=True)

try:
    AXIOM_PARSER = AxiomParser(model_config_path=MODEL_CONFIG_PATH)
    RULE_GENERATOR = RuleGenerator(model_config_path=MODEL_CONFIG_PATH, simulation_settings_path=SIM_SETTINGS_PATH)
    WORLD_COMPILER = WorldCompiler(simulation_settings_path=SIM_SETTINGS_PATH)
    FLOURISHING_EVALUATOR = FlourishingEvaluator(model_config_path=MODEL_CONFIG_PATH, axioms_config_path=AXIOMS_CONFIG_PATH)
    PARADOX_DETECTOR = ParadoxDetector(model_config_path=MODEL_CONFIG_PATH, axioms_config_path=AXIOMS_CONFIG_PATH)
    logger.info("All core AI components initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize one or more core AI components: {e}")
    logger.error("Please ensure all configuration files are valid and models can be loaded.")
    raise typer.Exit(code=1)

# --- CLI Commands ---

@app.command(name="generate", help="[bold green]Generate a new axiomatically aligned simulated world.[/bold green]")
def generate_world(
    world_name: str = typer.Argument(..., help="[bold]Name for your new simulated world.[/bold]"),
    axiom_file: str = typer.Option(AXIOMS_CONFIG_PATH, "--axioms", "-a", help="Path to the YAML file defining your world's axioms."),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format for the world configuration (json or yaml)."),
    output_dir: str = typer.Option(GENERATED_WORLDS_DIR, "--output-dir", "-o", help="Directory to save the generated world config.")
):
    """
    Generates a new axiomatically aligned simulated world based on specified axioms.
    The AI designs the foundational rules, parameters, and initial conditions.
    """
    logger.info(f"Initiating world generation for '{world_name}' using axioms from '{axiom_file}'...")
    try:
        # 1. Parse Axioms
        axiom_set = AXIOM_PARSER.parse_axioms(axiom_file)
        
        # 2. Generate Rules
        generated_rules = RULE_GENERATOR.generate_rules(axiom_set, world_name)
        
        # 3. Compile World Configuration
        compiled_config = WORLD_COMPILER.compile_world(generated_rules)
        
        # 4. Save World Configuration
        WORLD_COMPILER.save_world_config(compiled_config, output_dir, format=output_format)
        
        logger.success(f"[bold green]World '{world_name}' generated and saved successfully to {output_dir}/{world_name}.{output_format}[/bold green]")
        logger.info(f"Next step: You can now run 'opd evaluate --world-file {output_dir}/{world_name}.{output_format} --axiom-file {axiom_file}' to assess its alignment.")
    
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        typer.echo(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during world generation: {e}", exc_info=True)
        typer.echo(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command(name="evaluate", help="[bold blue]Evaluate an AI-designed world for axiomatic alignment and flourishing potential.[/bold blue]")
def evaluate_world(
    world_file: str = typer.Argument(..., help="[bold]Path to the compiled world configuration file (JSON or YAML).[/bold]"),
    axiom_file: str = typer.Option(AXIOMS_CONFIG_PATH, "--axioms", "-a", help="Path to the YAML file defining the axioms used for this world."),
    report_output_dir: str = typer.Option(EVALUATION_REPORTS_DIR, "--report-dir", "-o", help="Directory to save the evaluation report.")
):
    """
    Evaluates an AI-designed world configuration for its predicted axiom adherence,
    flourishing trajectory, and overall ethical alignment.
    """
    logger.info(f"Initiating evaluation for world from '{world_file}'...")
    try:
        # 1. Load Compiled World Configuration
        file_extension = os.path.splitext(world_file)[1].lower()
        if file_extension == '.json':
            with open(world_file, 'r') as f:
                compiled_config = json.load(f)
        elif file_extension == '.yaml' or file_extension == '.yml':
            with open(world_file, 'r') as f:
                compiled_config = yaml.safe_load(f)
        else:
            logger.error(f"Unsupported world file format: {file_extension}. Must be .json or .yaml.")
            raise ValueError("Unsupported world file format")
        
        # Create a mock GeneratedWorldRules object for ParadoxDetector
        # In a real scenario, we'd load this if it was saved separately or parse from compiled_config
        # For simplicity, reconstructing minimal object:
        mock_generated_rules = GeneratedWorldRules(
            world_name=compiled_config['world_metadata']['name'],
            rules=[GeneratedRule(id=r['id'], description=r['description'], type=r['type'], parameters=r['parameters'], dependencies=r.get('dependencies', []), axiom_influence=r.get('axiom_influence', {})) 
                   for category in compiled_config['generated_world_rules'].values() for r in category],
            rule_graph=None, # Not strictly needed for paradox detector mock here
            creation_timestamp=compiled_config['world_metadata']['creation_timestamp'],
            axioms_used_ids=compiled_config['world_metadata']['axioms_influencing_design']
        )
        logger.debug(f"Loaded world config for '{compiled_config['world_metadata']['name']}'.")

        # 2. Parse Axioms (again, to ensure we have the ParsedAxiom objects)
        axiom_set = AXIOM_PARSER.parse_axioms(axiom_file)

        # 3. Detect Paradoxes (Integrated step)
        paradox_report = PARADOX_DETECTOR.detect_paradoxes(mock_generated_rules, axiom_set)
        
        # 4. Evaluate Flourishing
        evaluation_report = FLOURISHING_EVALUATOR.evaluate_world(
            compiled_config, axiom_set, paradox_risk_score=paradox_report.total_paradox_risk_score
        )

        # 5. Save Evaluation Report
        report_filename = os.path.join(report_output_dir, f"{evaluation_report.world_name}_report_{TIMESTAMP}.txt")
        with open(report_filename, 'w') as f:
            f.write(f"--- World Evaluation Report for: {evaluation_report.world_name} ---\n")
            f.write(f"Evaluation Timestamp: {evaluation_report.evaluation_timestamp}\n")
            f.write(f"Overall Flourishing Score: {evaluation_report.overall_flourishing_score:.2f}\n")
            f.write(f"Paradox Risk Score: {evaluation_report.paradox_risk_score:.2f}\n")
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
        
        logger.success(f"[bold blue]Evaluation report for '{evaluation_report.world_name}' generated successfully: {report_filename}[/bold blue]")
    
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        typer.echo(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"Error: {e}")
        typer.echo(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during world evaluation: {e}", exc_info=True)
        typer.echo(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command(name="list-axioms", help="[bold yellow]Display the currently loaded axiomatic principles.[/bold yellow]")
def list_axioms(
    axiom_file: str = typer.Option(AXIOMS_CONFIG_PATH, "--axioms", "-a", help="Path to the YAML file defining your world's axioms.")
):
    """
    Parses and displays the axiomatic principles from the specified file.
    """
    logger.info(f"Listing axioms from '{axiom_file}'...")
    try:
        axiom_set = AXIOM_PARSER.parse_axioms(axiom_file)
        typer.echo(f"\n[bold yellow]--- Loaded Axiomatic Principles from {axiom_file} ---[/bold yellow]")
        for axiom in axiom_set.axioms:
            typer.echo(f"[bold white]ID:[/] {axiom.id}")
            typer.echo(f"  [bold white]Principle:[/] {axiom.principle_text}")
            typer.echo(f"  [bold white]Priority:[/] {axiom.priority} (lower is higher priority)")
            typer.echo(f"  [bold white]Type:[/] {axiom.type}")
            typer.echo(f"  [bold white]Keywords:[/] {', '.join(axiom.keywords)}")
            if axiom.enforcement_strategy:
                typer.echo(f"  [bold white]Enforcement Strategy:[/] {axiom.enforcement_strategy}")
            typer.echo("-" * 60)
        logger.success(f"[bold green]Successfully listed {len(axiom_set.axioms)} axioms.[/bold green]")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        typer.echo(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing axioms: {e}", exc_info=True)
        typer.echo(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)

@app.command(name="clean", help="[bold red]Clean up generated data and reports.[/bold red]")
def clean_data(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion without prompt.")
):
    """
    Deletes all files in `data/generated_worlds/` and `data/evaluation_reports/`.
    """
    if not confirm:
        typer.confirm(
            "[bold red]Are you sure you want to delete all generated world configurations and evaluation reports?[/bold red]",
            abort=True,
        )
    
    logger.info("Initiating data cleanup...")
    try:
        shutil.rmtree(GENERATED_WORLDS_DIR, ignore_errors=True)
        shutil.rmtree(EVALUATION_REPORTS_DIR, ignore_errors=True)
        os.makedirs(GENERATED_WORLDS_DIR, exist_ok=True) # Recreate empty directories
        os.makedirs(EVALUATION_REPORTS_DIR, exist_ok=True)
        logger.success(f"[bold green]Cleaned up '{GENERATED_WORLDS_DIR}' and '{EVALUATION_REPORTS_DIR}' successfully.[/bold green]")
    except Exception as e:
        logger.error(f"An error occurred during cleanup: {e}", exc_info=True)
        typer.echo(f"[bold red]An error occurred during cleanup: {e}[/bold red]")
        raise typer.Exit(code=1)


def main():
    app()

if __name__ == "__main__":
    main()
