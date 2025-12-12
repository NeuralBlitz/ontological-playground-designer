# ontological-playground-designer/src/interfaces/api.py

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import uvicorn

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Import core AI components (these will be initialized globally or via dependency injection)
from src.core.axiom_parser import AxiomParser, AxiomSet, ParsedAxiom
from src.core.rule_generator import RuleGenerator, GeneratedWorldRules, GeneratedRule
from src.core.world_compiler import WorldCompiler
from src.core.flourishing_evaluator import FlourishingEvaluator, WorldEvaluationReport, EvaluationMetric
from src.core.paradox_detector import ParadoxDetector, ParadoxDetectionReport, DetectedParadox

# Setup logging for the API
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
API_LOG_FILE = os.path.join(LOG_DIR, "opd_api.log")
setup_logging(log_level="INFO", log_file=API_LOG_FILE)


app = FastAPI(
    title="Ontological Playground Designer API",
    description="API for generating and evaluating axiomatically aligned simulated worlds.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Global AI Component Initialization ---
# In a production setup, these would be managed by a dependency injection system
# or loaded more robustly. For this placeholder, we'll mimic the CLI's global load.
#
# Configuration paths (relative to project root)
AXIOMS_CONFIG_PATH = "config/axioms.yaml"
MODEL_CONFIG_PATH = "config/model_config.yaml"
SIM_SETTINGS_PATH = "config/simulation_settings.yaml"

# Output paths
GENERATED_WORLDS_DIR = "data/generated_worlds"
EVALUATION_REPORTS_DIR = "data/evaluation_reports"

try:
    AXIOM_PARSER = AxiomParser(model_config_path=MODEL_CONFIG_PATH)
    RULE_GENERATOR = RuleGenerator(model_config_path=MODEL_CONFIG_PATH, simulation_settings_path=SIM_SETTINGS_PATH)
    WORLD_COMPILER = WorldCompiler(simulation_settings_path=SIM_SETTINGS_PATH)
    FLOURISHING_EVALUATOR = FlourishingEvaluator(model_config_path=MODEL_CONFIG_PATH, axioms_config_path=AXIOMS_CONFIG_PATH)
    PARADOX_DETECTOR = ParadoxDetector(model_config_path=MODEL_CONFIG_PATH, axioms_config_path=AXIOMS_CONFIG_PATH)
    logger.info("API: All core AI components initialized successfully.")
except Exception as e:
    logger.error(f"API: Failed to initialize one or more core AI components: {e}")
    logger.error("API: Please ensure all configuration files are valid and models can be loaded.")
    raise RuntimeError("API initialization failed") # Let the app crash on init failure


# --- Pydantic Models for API Request/Response ---

class AxiomDefinition(BaseModel):
    id: str = Field(..., description="Unique identifier for the axiom.")
    principle: str = Field(..., description="A clear, concise statement of the axiom's core principle.")
    priority: int = Field(99, description="Relative importance (lower number = higher priority).")
    type: str = Field("generic", description="Category of the axiom (e.g., ethical, environmental).")
    keywords: List[str] = Field(default_factory=list, description="Keywords for semantic parsing.")
    enforcement_strategy: Optional[str] = Field(None, description="High-level guidance for enforcement.")

class AxiomSetRequest(BaseModel):
    world_name: str = Field(..., description="Name for the new simulated world.")
    axioms: List[AxiomDefinition] = Field(..., description="List of axiomatic principles for the world design.")

class GeneratedWorldResponse(BaseModel):
    world_name: str
    creation_timestamp: str
    output_format: str
    output_file: str
    message: str

class WorldEvaluationRequest(BaseModel):
    world_name: str = Field(..., description="Name of the world to evaluate.")
    world_config: Dict[str, Any] = Field(..., description="The full compiled world configuration (JSON content).")
    axioms: List[AxiomDefinition] = Field(..., description="List of axioms used to design this world.")

class WorldEvaluationResponse(BaseModel):
    world_name: str
    evaluation_timestamp: str
    overall_flourishing_score: float
    axiom_adherence_scores: Dict[str, float]
    predicted_metrics: List[EvaluationMetric]
    paradox_risk_score: float
    detected_paradoxes: List[DetectedParadox]
    recommendations: List[str]
    message: str

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Checks if the API and core components are operational."""
    logger.info("API health check requested.")
    if AXIOM_PARSER and RULE_GENERATOR and WORLD_COMPILER and FLOURISHING_EVALUATOR and PARADOX_DETECTOR:
        return {"status": "operational", "message": "Ontological Playground Designer API is ready."}
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Core components not initialized.")

@app.post("/generate_world", response_model=GeneratedWorldResponse, status_code=status.HTTP_201_CREATED)
async def api_generate_world(request: AxiomSetRequest):
    """
    Generates a new axiomatically aligned simulated world based on provided axioms.
    """
    logger.info(f"API: Generating world '{request.world_name}'...")
    try:
        # Convert Pydantic AxiomDefinition to ParsedAxiom
        parsed_axioms_list = [
            ParsedAxiom(
                id=a.id,
                principle_text=a.principle,
                priority=a.priority,
                type=a.type,
                keywords=a.keywords,
                enforcement_strategy=a.enforcement_strategy,
                embedding=AXIOM_PARSER._get_embedding(a.principle) # Generate embedding on the fly
            ) for a in request.axioms
        ]
        # Sort axioms by priority (lower number = higher priority) for consistency
        parsed_axioms_list.sort(key=lambda x: x.priority)
        axiom_set = AxiomSet(axioms=parsed_axioms_list)
        
        generated_rules = RULE_GENERATOR.generate_rules(axiom_set, request.world_name)
        compiled_config = WORLD_COMPILER.compile_world(generated_rules)
        
        output_file_name = f"{request.world_name}.json" # Default to JSON for API output
        output_full_path = os.path.join(GENERATED_WORLDS_DIR, output_file_name)
        
        # Save to file
        with open(output_full_path, 'w') as f:
            json.dump(compiled_config, f, indent=4)

        logger.success(f"API: World '{request.world_name}' generated and saved successfully.")
        return GeneratedWorldResponse(
            world_name=request.world_name,
            creation_timestamp=generated_rules.creation_timestamp,
            output_format="json",
            output_file=output_full_path,
            message="World designed successfully."
        )
    except Exception as e:
        logger.error(f"API: Error generating world '{request.world_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/evaluate_world", response_model=WorldEvaluationResponse, status_code=status.HTTP_200_OK)
async def api_evaluate_world(request: WorldEvaluationRequest):
    """
    Evaluates an AI-designed world for axiomatic alignment and flourishing potential.
    """
    logger.info(f"API: Evaluating world '{request.world_name}'...")
    try:
        # Convert Pydantic AxiomDefinition to ParsedAxiom
        parsed_axioms_list = [
            ParsedAxiom(
                id=a.id,
                principle_text=a.principle,
                priority=a.priority,
                type=a.type,
                keywords=a.keywords,
                enforcement_strategy=a.enforcement_strategy,
                embedding=AXIOM_PARSER._get_embedding(a.principle) # Generate embedding on the fly
            ) for a in request.axioms
        ]
        # Sort axioms by priority for consistency
        parsed_axioms_list.sort(key=lambda x: x.priority)
        axiom_set = AxiomSet(axioms=parsed_axioms_list)

        # Reconstruct minimal GeneratedWorldRules for ParadoxDetector
        mock_generated_rules = GeneratedWorldRules(
            world_name=request.world_name,
            rules=[GeneratedRule(id=r['id'], description=r['description'], type=r['type'], parameters=r['parameters'], dependencies=r.get('dependencies', []), axiom_influence=r.get('axiom_influence', {})) 
                   for category in request.world_config.get('generated_world_rules', {}).values() for r in category],
            rule_graph=None,
            creation_timestamp=request.world_config.get('world_metadata', {}).get('creation_timestamp', datetime.datetime.now().isoformat()),
            axioms_used_ids=request.world_config.get('world_metadata', {}).get('axioms_influencing_design', [])
        )
        
        paradox_report = PARADOX_DETECTOR.detect_paradoxes(mock_generated_rules, axiom_set)
        
        evaluation_report = FLOURISHING_EVALUATOR.evaluate_world(
            request.world_config, axiom_set, paradox_risk_score=paradox_report.total_paradox_risk_score
        )
        
        logger.success(f"API: Evaluation for '{request.world_name}' completed successfully.")
        return WorldEvaluationResponse(
            world_name=evaluation_report.world_name,
            evaluation_timestamp=evaluation_report.evaluation_timestamp,
            overall_flourishing_score=evaluation_report.overall_flourishing_score,
            axiom_adherence_scores=evaluation_report.axiom_adherence_scores,
            predicted_metrics=evaluation_report.predicted_metrics,
            paradox_risk_score=evaluation_report.paradox_risk_score,
            detected_paradoxes=paradox_report.detected_paradoxes, # Include full paradox details
            recommendations=evaluation_report.recommendations,
            message="World evaluated successfully."
        )
    except Exception as e:
        logger.error(f"API: Error evaluating world '{request.world_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- To run the API server (for development/testing) ---
# You would typically run this using `uvicorn src.interfaces.api:app --reload`
if __name__ == "__main__":
    logger.info("Starting API server for Ontological Playground Designer...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
