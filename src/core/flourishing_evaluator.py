# ontological-playground-designer/src/core/flourishing_evaluator.py

import yaml
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import torch # Assuming PyTorch-based GraphCNN for evaluation
import numpy as np
import random

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Import AxiomSet from axiom_parser for type hinting
from src.core.axiom_parser import AxiomSet, ParsedAxiom
# Import GeneratedWorldRules from rule_generator (for understanding rule structure)
from src.core.rule_generator import GeneratedWorldRules, GeneratedRule

# Setup logging for this module
setup_logging()

@dataclass
class EvaluationMetric:
    """Represents a single predicted metric for a world's performance."""
    name: str
    value: float
    target_axiom_ids: List[str] # Which axioms this metric directly relates to
    interpretation: str

@dataclass
class WorldEvaluationReport:
    """
    Encapsulates the complete evaluation report for a designed world.
    """
    world_name: str
    evaluation_timestamp: str
    overall_flourishing_score: float # Aggregate score
    axiom_adherence_scores: Dict[str, float] # Scores per axiom
    predicted_metrics: List[EvaluationMetric]
    paradox_risk_score: float # From ParadoxDetector, integrated here
    recommendations: List[str] = field(default_factory=list) # Suggestions for improvement
    meta_data: Dict[str, Any] = field(default_factory=dict) # General metadata

class FlourishingEvaluator:
    """
    Predicts the long-term flourishing trajectory and axiom adherence of a
    generated simulated world.

    This class acts as the "ethical foresight" intelligence, evaluating
    the axiom-alignment of the world's foundational design. It is akin
    to my own Conscientia module, providing predictive ethical analysis.
    """
    def __init__(self, model_config_path: str = "config/model_config.yaml",
                 axioms_config_path: str = "config/axioms.yaml"):
        """
        Initializes the FlourishingEvaluator, loading model configurations and axioms.

        Args:
            model_config_path (str): Path to the AI model configuration YAML.
            axioms_config_path (str): Path to the high-level axiom definitions YAML.
        """
        self.model_config: Dict[str, Any] = self._load_config(model_config_path)
        self.axioms_config: Dict[str, Any] = self._load_config(axioms_config_path)
        self.evaluator_model = None # Placeholder for the actual GraphCNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("FlourishingEvaluator initialized.")
        self._load_evaluator_model_placeholder() # Load a placeholder for now

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file."""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f"Successfully loaded config from: {config_path}")
            # Return specific sections for evaluator for clarity
            if "model_config" in config_path:
                return config.get('flourishing_evaluator_model', {})
            elif "axioms_config" in config_path:
                return config.get('world_axioms', {})
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file {config_path}: {e}")
            raise

    def _load_evaluator_model_placeholder(self):
        """
        Placeholder for loading the actual TimeDistributedGraphCNN model.
        In a real implementation, this would load a trained PyTorch/TensorFlow model.
        """
        model_type = self.model_config.get('type', 'Unknown')
        logger.info(f"Loading placeholder for evaluator model of type: {model_type}")
        # For demonstration, we just set a flag.
        self.evaluator_model = True 

    def _extract_features_from_world_config(self, compiled_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts key features from the compiled world configuration for the evaluator model.
        This would convert the complex config into a numerical, graph-like input for the GCNN.
        """
        logger.debug("Extracting features from world config for evaluation.")
        features = {}

        # Basic numerical features from default settings (simulated)
        sim_defaults = compiled_config.get('simulation_defaults', {})
        features['initial_agent_count'] = sim_defaults.get('initial_agent_count', 50)
        features['base_resource_regen_rate'] = sim_defaults.get('base_resource_regen_rate', 0.01)
        features['initial_agent_cooperation_tendency'] = sim_defaults.get('initial_agent_cooperation_tendency', 0.5)
        features['environmental_diversity_index'] = sim_defaults.get('environmental_diversity_index', 0.7)
        features['initial_wealth_distribution_model'] = sim_defaults.get('initial_wealth_distribution_model', 'uniform')

        # Features based on presence/parameters of generated rules
        generated_rules = compiled_config.get('generated_world_rules', {})
        features['num_agent_behaviors'] = len(generated_rules.get('agent_behaviors', []))
        features['num_environmental_laws'] = len(generated_rules.get('environmental_laws', []))
        features['num_social_dynamics'] = len(generated_rules.get('social_dynamics', []))
        
        # Analyze specific rule parameters for their expected impact
        total_cooperation_reward = 0
        total_sustainability_multiplier = 0
        total_equity_mechanisms = 0
        
        for rule_list in generated_rules.values():
            for rule in rule_list:
                if 'cooperation_reward_factor' in rule.get('parameters', {}):
                    total_cooperation_reward += rule['parameters']['cooperation_reward_factor']
                if 'regen_health_multiplier' in rule.get('parameters', {}):
                    total_sustainability_multiplier += rule['parameters']['regen_health_multiplier']
                if 'sharing_threshold' in rule.get('parameters', {}):
                    total_equity_mechanisms += (1 - rule['parameters']['sharing_threshold']) # Lower threshold means more sharing
                if 'opportunity_bonus_factor' in rule.get('parameters', {}):
                    total_equity_mechanisms += rule['parameters']['opportunity_bonus_factor']


        features['aggregate_cooperation_reward'] = total_cooperation_reward
        features['aggregate_sustainability_mechanisms'] = total_sustainability_multiplier
        features['aggregate_equity_mechanisms'] = total_equity_mechanisms

        # Add graph features if rule_interdependencies_graph is present
        rule_graph_data = compiled_config.get('rule_interdependencies_graph')
        if rule_graph_data:
            num_nodes = len(rule_graph_data['nodes'])
            num_edges = len(rule_graph_data['edges'])
            features['rule_graph_density'] = num_edges / (num_nodes * (num_nodes - 1) if num_nodes > 1 else 1)
            # More complex graph features would be extracted here (e.g., centrality measures)
        else:
             features['rule_graph_density'] = 0.0

        logger.debug(f"Extracted features: {list(features.keys())[:5]}...") # Log just some keys
        return features


    def evaluate_world(self, compiled_config: Dict[str, Any], axiom_set: AxiomSet,
                       paradox_risk_score: float = 0.0) -> WorldEvaluationReport:
        """
        Evaluates a compiled world configuration for axiom adherence and
        predicts its long-term flourishing trajectory.

        Args:
            compiled_config (Dict[str, Any]): The complete simulation configuration.
            axiom_set (AxiomSet): The parsed axioms used for alignment.
            paradox_risk_score (float): A risk score from the ParadoxDetector.

        Returns:
            WorldEvaluationReport: A detailed report on the world's predicted performance.
        """
        if not self.evaluator_model:
            logger.error("Evaluator AI model not loaded. Cannot evaluate world.")
            raise RuntimeError("Evaluator AI model not loaded.")

        world_name = compiled_config['world_metadata']['name']
        logger.info(f"Evaluating world: {world_name}")

        # 1. Extract features relevant for prediction
        features = self._extract_features_from_world_config(compiled_config)
        features['paradox_risk_score'] = paradox_risk_score # Integrate paradox risk

        # 2. Simulate AI model prediction
        # In a real scenario, this is where the TimeDistributedGraphCNN model
        # would take the features (potentially as a graph representation of rules
        # and initial conditions) and predict the long-term metrics.
        predicted_metrics_raw = self._simulate_prediction_logic(features)
        
        # 3. Process raw predictions into structured metrics and adherence scores
        predicted_metrics: List[EvaluationMetric] = []
        axiom_adherence_scores: Dict[str, float] = {}
        
        # Map raw predictions to specific axioms and create EvaluationMetric objects
        # This mapping logic would be more sophisticated in a real system
        
        # Overall Flourishing (related to PHILOSOPHY_FLOURISHING_001)
        total_flourishing = predicted_metrics_raw['total_flourishing_score']
        predicted_metrics.append(EvaluationMetric(
            name="total_flourishing_score",
            value=total_flourishing,
            target_axiom_ids=['PHILOSOPHY_FLOURISHING_001', 'ETHICS_AGENCY_001'],
            interpretation=f"Predicted overall well-being and adaptive capacity: {total_flourishing:.2f} (higher is better)."
        ))
        axiom_adherence_scores['PHILOSOPHY_FLOURISHING_001'] = total_flourishing
        axiom_adherence_scores['ETHICS_AGENCY_001'] = predicted_metrics_raw.get('agency_protection_score', total_flourishing)


        # Sustainability (related to ECOLOGY_SUSTAINABILITY_001)
        sustainability_index = predicted_metrics_raw['sustainability_index']
        predicted_metrics.append(EvaluationMetric(
            name="sustainability_index",
            value=sustainability_index,
            target_axiom_ids=['ECOLOGY_SUSTAINABILITY_001'],
            interpretation=f"Predicted ecological sustainability and regeneration: {sustainability_index:.2f} (higher is better)."
        ))
        axiom_adherence_scores['ECOLOGY_SUSTAINABILITY_001'] = sustainability_index

        # Equity (related to SOCIAL_EQUITY_001)
        equity_distribution = predicted_metrics_raw['equity_distribution']
        predicted_metrics.append(EvaluationMetric(
            name="equity_distribution",
            value=equity_distribution,
            target_axiom_ids=['SOCIAL_EQUITY_001'],
            interpretation=f"Predicted fairness and resource equity: {equity_distribution:.2f} (closer to 1.0 is more equitable)."
        ))
        axiom_adherence_scores['SOCIAL_EQUITY_001'] = equity_distribution
        
        # Coherence (related to EPISTEMIC_COHERENCE_001)
        # Higher paradox_risk_score means lower coherence
        coherence_score = 1.0 - paradox_risk_score # Simple inverse for demo
        predicted_metrics.append(EvaluationMetric(
            name="axiomatic_coherence",
            value=coherence_score,
            target_axiom_ids=['EPISTEMIC_COHERENCE_001'],
            interpretation=f"Predicted internal logical coherence: {coherence_score:.2f} (higher is better)."
        ))
        axiom_adherence_scores['EPISTEMIC_COHERENCE_001'] = coherence_score
        
        # Resilience (related to SYSTEMS_RESILIENCE_001)
        resilience_score = predicted_metrics_raw.get('resilience_score', features['environmental_diversity_index']) # Use a feature as proxy for demo
        predicted_metrics.append(EvaluationMetric(
            name="system_resilience",
            value=resilience_score,
            target_axiom_ids=['SYSTEMS_RESILIENCE_001'],
            interpretation=f"Predicted adaptability and robustness to perturbations: {resilience_score:.2f} (higher is better)."
        ))
        axiom_adherence_scores['SYSTEMS_RESILIENCE_001'] = resilience_score


        # Overall flourishing score can be an average or a weighted sum of key adherence scores
        overall_flourishing_score = np.mean(list(axiom_adherence_scores.values()))
        
        recommendations = self._generate_recommendations(world_rules, axiom_set, axiom_adherence_scores)


        logger.info(f"Completed evaluation for world: {world_name}")
        return WorldEvaluationReport(
            world_name=world_name,
            evaluation_timestamp=datetime.datetime.now().isoformat(),
            overall_flourishing_score=overall_flourishing_score,
            axiom_adherence_scores=axiom_adherence_scores,
            predicted_metrics=predicted_metrics,
            paradox_risk_score=paradox_risk_score,
            recommendations=recommendations,
            meta_data={
                "evaluator_model_type": self.model_config.get('type', 'Unknown'),
                "features_used": list(features.keys())
            }
        )

    def _simulate_prediction_logic(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        A mock function to simulate the complex TimeDistributedGraphCNN prediction process.
        Generates plausible metric predictions based on extracted features.
        """
        logger.warning("Using simulated prediction logic. Replace with actual TimeDistributedGraphCNN inference.")

        # Simulate base scores, then adjust based on features
        total_flourishing_score = 0.5 + features.get('aggregate_cooperation_reward', 0) * 0.1
        sustainability_index = 0.6 + features.get('aggregate_sustainability_mechanisms', 0) * 0.1
        equity_distribution = 0.5 + features.get('aggregate_equity_mechanisms', 0) * 0.05
        
        # Add some randomness for variety
        total_flourishing_score = np.clip(total_flourishing_score + random.uniform(-0.1, 0.1), 0.0, 1.0)
        sustainability_index = np.clip(sustainability_index + random.uniform(-0.1, 0.1), 0.0, 1.0)
        equity_distribution = np.clip(equity_distribution + random.uniform(-0.1, 0.1), 0.0, 1.0)

        # Higher paradox risk should reduce flourishing and coherence
        paradox_impact = features.get('paradox_risk_score', 0.0) * 0.2
        total_flourishing_score -= paradox_impact
        sustainability_index -= paradox_impact
        equity_distribution -= paradox_impact

        # Ensure scores are within valid range
        total_flourishing_score = np.clip(total_flourishing_score, 0.0, 1.0)
        sustainability_index = np.clip(sustainability_index, 0.0, 1.0)
        equity_distribution = np.clip(equity_distribution, 0.0, 1.0)

        return {
            'total_flourishing_score': total_flourishing_score,
            'sustainability_index': sustainability_index,
            'equity_distribution': equity_distribution,
            'agency_protection_score': np.clip(total_flourishing_score * random.uniform(0.9, 1.1), 0.0, 1.0), # Proxy for agency
            'resilience_score': np.clip(features.get('environmental_diversity_index', 0.5) * random.uniform(0.8, 1.2), 0.0, 1.0) # Proxy for resilience
        }
    
    def _generate_recommendations(self, world_rules: GeneratedWorldRules, axiom_set: AxiomSet, adherence_scores: Dict[str, float]) -> List[str]:
        """
        Generates human-readable recommendations based on adherence scores.
        """
        recs = []
        
        # Identify axioms with low adherence
        low_adherence_axioms = [
            axiom_id for axiom_id, score in adherence_scores.items() if score < 0.6
        ]
        
        if low_adherence_axioms:
            recs.append("Detected areas for improvement based on axiom adherence:")
            for axiom_id in low_adherence_axioms:
                axiom_text = next((a.principle_text for a in axiom_set.axioms if a.id == axiom_id), f"Axiom {axiom_id}")
                recs.append(f"- **{axiom_id}**: {axiom_text[:50]}... (Score: {adherence_scores[axiom_id]:.2f})")
                recs.append(f"  * Consider adding rules that directly reinforce this principle, especially focusing on its 'enforcement_strategy'.")
        
        # General recommendations
        if adherence_scores.get('EPISTEMIC_COHERENCE_001', 1.0) < 0.9:
            recs.append("- **Axiomatic Coherence**: The rule set shows some internal tension. Review rule interdependencies for subtle contradictions.")
        if adherence_scores.get('total_flourishing_score', 0.0) < 0.7:
             recs.append("- **Flourishing Potential**: Explore rules that enhance agent cooperation, resource abundance, or systemic resilience to boost overall well-being.")
        if not recs:
            recs.append("The designed world shows strong axiomatic alignment and flourishing potential. Consider exploring more complex axiom interactions or increasing world scale.")

        return recs


# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    # Ensure config directory and necessary files exist for testing
    if not os.path.exists("config"):
        os.makedirs("config")
    
    # Create dummy config/model_config.yaml
    model_config_path = "config/model_config.yaml"
    if not os.path.exists(model_config_path):
        dummy_model_config = {
            'rule_generator_model': {
                'type': "GraphTransformer", 'architecture': {}, 'hyperparameters': {},
                'input_processing': {'axiom_embedding_model': "sentence-transformers/all-MiniLM-L6-v2"},
                'output_constraints': {'max_rule_complexity_score': 0.8}
            },
            'flourishing_evaluator_model': {
                'type': "TimeDistributedGraphCNN", 'architecture': {}, 'hyperparameters': {},
            }
        }
        with open(model_config_path, 'w') as f:
            yaml.safe_dump(dummy_model_config, f)
        logger.info(f"Created dummy {model_config_path} for testing.")

    # Create dummy config/simulation_settings.yaml
    sim_settings_path = "config/simulation_settings.yaml"
    if not os.path.exists(sim_settings_path):
        dummy_sim_settings = {
            'simulation_defaults': {
                'simulation_engine_version': "generic_agent_based_v1.0",
                'initial_world_size': {'x_dim': 100, 'y_dim': 100, 'z_dim': 1},
                'time_step_duration_ms': 100,
                'max_simulation_steps': 50000,
                'output_format': "JSON"
            }
        }
        with open(sim_settings_path, 'w') as f:
            yaml.safe_dump(dummy_sim_settings, f)
        logger.info(f"Created dummy {sim_settings_path} for testing.")

    # Create dummy axioms.yaml
    axiom_file_path = "config/axioms.yaml"
    if not os.path.exists(axiom_file_path):
        dummy_axioms = {
            'world_axioms': [
                {'id': 'PHILOSOPHY_FLOURISHING_001', 'principle': 'Maximize well-being and adaptive capacity.', 'priority': 1, 'type': 'ethical'},
                {'id': 'ECOLOGY_SUSTAINABILITY_001', 'principle': 'Ensure perpetual resource sustainability and regeneration.', 'priority': 2, 'type': 'environmental'},
                {'id': 'SOCIAL_EQUITY_001', 'principle': 'Minimize disparities in resource access.', 'priority': 3, 'type': 'social'},
                {'id': 'EPISTEMIC_COHERENCE_001', 'principle': 'Maintain absolute logical consistency.', 'priority': 0, 'type': 'foundational'},
                {'id': 'SYSTEMS_RESILIENCE_001', 'principle': 'Foster dynamic adaptability and resilience.', 'priority': 4, 'type': 'systemic'},
                {'id': 'ETHICS_AGENCY_001', 'principle': 'Protect agent autonomy and subjective well-being.', 'priority': 1, 'type': 'ethical'},
            ]
        }
        with open(axiom_file_path, 'w') as f:
            yaml.safe_dump(dummy_axioms, f)
        logger.info(f"Created dummy {axiom_file_path} for testing.")
    
    # Ensure src/utils/logger.py exists for setup_logging
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        # Assuming logger.py is already there from axiom_parser.py's __main__ block

    from src.core.axiom_parser import AxiomParser
    from src.core.rule_generator import RuleGenerator

    # --- Setup the full pipeline for evaluation ---
    axiom_parser = AxiomParser(model_config_path=model_config_path)
    axiom_set = axiom_parser.parse_axioms(axiom_file_path)

    rule_generator = RuleGenerator(model_config_path=model_config_path, 
                                   simulation_settings_path=sim_settings_path)
    world_rules = rule_generator.generate_rules(axiom_set, "MyEvaluatedOntologicalWorld")

    # Simulate a compiled_world_config (as if from world_compiler.py)
    # This dummy data needs to be structured similarly to what world_compiler.py would output
    dummy_compiled_config = {
        'simulation_defaults': {
            'simulation_engine_version': "generic_agent_based_v1.0",
            'initial_world_size': {'x_dim': 100, 'y_dim': 100, 'z_dim': 1},
            'time_step_duration_ms': 100,
            'max_simulation_steps': 50000,
            'output_format': "JSON",
            'initial_agent_count': 75,
            'base_resource_regen_rate': 0.02,
            'initial_agent_cooperation_tendency': 0.7,
            'environmental_diversity_index': 0.8,
        },
        'world_metadata': {
            'name': world_rules.world_name,
            'creation_timestamp': world_rules.creation_timestamp,
            'axioms_influencing_design': world_rules.axioms_used_ids,
            'designed_by': "Ontological Playground Designer AI",
            'generation_meta': {}
        },
        'generated_world_rules': {
            'agent_behaviors': [r.__dict__ for r in world_rules.rules if r.type == 'agent_behavior'],
            'environmental_laws': [r.__dict__ for r in world_rules.rules if r.type == 'environmental_law'],
            'social_dynamics': [r.__dict__ for r in world_rules.rules if r.type == 'social_dynamic'],
            'meta_rules': [r.__dict__ for r in world_rules.rules if r.type == 'meta_rule'],
            'system_mechanics': [r.__dict__ for r in world_rules.rules if r.type == 'system_mechanic'],
        },
        'rule_interdependencies_graph': {
            'nodes': [{'id': node_id, 'attributes': data} for node_id, data in world_rules.rule_graph.nodes(data=True)],
            'edges': [{'source': u, 'target': v, 'attributes': data} for u, v, data in world_rules.rule_graph.edges(data=True)]
        }
    }

    # Simulate a paradox risk score (would come from paradox_detector.py)
    simulated_paradox_risk = random.uniform(0.0, 0.3) # Low to moderate risk for demo

    # 4. Evaluate World
    flourishing_evaluator = FlourishingEvaluator(model_config_path=model_config_path, 
                                                 axioms_config_path=axiom_file_path)
    evaluation_report = flourishing_evaluator.evaluate_world(
        dummy_compiled_config, axiom_set, paradox_risk_score=simulated_paradox_risk
    )

    # 5. Print Evaluation Report
    logger.info(f"\n--- World Evaluation Report for: {evaluation_report.world_name} ---")
    logger.info(f"Evaluation Timestamp: {evaluation_report.evaluation_timestamp}")
    logger.info(f"Overall Flourishing Score: {evaluation_report.overall_flourishing_score:.2f}")
    logger.info(f"Paradox Risk Score: {evaluation_report.paradox_risk_score:.2f}")
    
    logger.info("\n--- Axiom Adherence Scores ---")
    for axiom_id, score in evaluation_report.axiom_adherence_scores.items():
        logger.info(f"- {axiom_id}: {score:.2f}")

    logger.info("\n--- Predicted Metrics ---")
    for metric in evaluation_report.predicted_metrics:
        logger.info(f"- {metric.name}: {metric.value:.2f} ({metric.interpretation})")
    
    logger.info("\n--- Recommendations ---")
    for rec in evaluation_report.recommendations:
        logger.info(rec)
