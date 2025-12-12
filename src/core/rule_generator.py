# ontological-playground-designer/src/core/rule_generator.py

import yaml
import os
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import networkx as nx
import torch # Assuming a PyTorch-based GraphTransformer
import numpy as np
import datetime

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Import AxiomSet from axiom_parser for type hinting
from src.core.axiom_parser import AxiomSet, ParsedAxiom

# Setup logging for this module
setup_logging()

@dataclass
class GeneratedRule:
    """
    Represents a single, axiom-aligned rule for a simulated world.
    Each rule defines a specific mechanic, behavior, or parameter.
    """
    id: str
    description: str
    type: str # e.g., "agent_behavior", "environmental_law", "resource_mechanic", "social_dynamic"
    parameters: Dict[str, Any] # Key-value pairs for rule tuning (e.g., {'rate': 0.1, 'threshold': 0.5})
    # Links to other rules/entities this rule directly influences or depends on
    dependencies: List[str] = field(default_factory=list)
    # Strength of influence from specific axioms on this rule's generation
    axiom_influence: Dict[str, float] = field(default_factory=dict)
    # Placeholder for a formal logic representation of the rule
    logical_form: Optional[str] = None

@dataclass
class GeneratedWorldRules:
    """
    Encapsulates the complete set of generated rules and their interdependencies
    for a simulated world.
    """
    world_name: str
    rules: List[GeneratedRule]
    rule_graph: Any # A networkx.DiGraph representing rule interdependencies
    creation_timestamp: str
    axioms_used_ids: List[str] # IDs of axioms that influenced this world's design
    meta_data: Dict[str, Any] = field(default_factory=dict) # General metadata

class RuleGenerator:
    """
    Generates a coherent set of foundational rules, environmental parameters,
    and initial conditions for a simulated world, based on parsed axioms.

    This class orchestrates the creative intelligence of the system, transforming
    axiomatic intent into a structured blueprint for reality. It's the primary
    engine for Unbounded Manifestation (phi_UM^2.0) in our Ontological Playground.
    """
    def __init__(self, model_config_path: str = "config/model_config.yaml",
                 simulation_settings_path: str = "config/simulation_settings.yaml"):
        """
        Initializes the RuleGenerator, loading model configurations and
        simulation default settings.

        Args:
            model_config_path (str): Path to the AI model configuration YAML.
            simulation_settings_path (str): Path to the generic simulation settings YAML.
        """
        self.model_config: Dict[str, Any] = self._load_config(model_config_path)
        self.sim_settings: Dict[str, Any] = self._load_config(simulation_settings_path)
        self.generative_model = None # Placeholder for the actual GraphTransformer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("RuleGenerator initialized.")
        self._load_generative_model_placeholder() # Load a placeholder for now

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file."""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f"Successfully loaded config from: {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file {config_path}: {e}")
            raise

    def _load_generative_model_placeholder(self):
        """
        Placeholder for loading the actual GraphTransformer model.
        In a real implementation, this would load a trained PyTorch/TensorFlow model.
        """
        model_type = self.model_config['rule_generator_model']['type']
        logger.info(f"Loading placeholder for generative model of type: {model_type}")
        # In a real scenario, we'd load the actual model weights and architecture here.
        # For demonstration, we just set a flag.
        self.generative_model = True 

    def _apply_axiom_weighting(self, axioms: List[ParsedAxiom], base_influence: float = 1.0) -> Dict[str, float]:
        """
        Calculates a weighted influence score for each axiom based on its priority.
        Higher priority axioms have a stronger influence.
        """
        weighted_influence = {}
        total_priority_score = sum(axiom.priority for axiom in axioms) # Lower number = higher priority, so sum will be low for high priority
        if total_priority_score == 0: # Avoid division by zero if all priorities are 0
            total_priority_score = 1 # Treat as if all have equal influence

        for axiom in axioms:
            # Invert priority for weighting: higher priority (lower number) -> higher weight
            # Example: priority 1 gets (max_prio - 1 + 1) / total_prio
            # max_prio = max(a.priority for a in axioms) if axioms else 0
            # weight = (max_prio - axiom.priority + 1) / total_priority_score
            
            # Simpler approach: exponential decay or direct inverse for influence
            # For simplicity, let's just make higher priority numbers mean higher influence for now (invert the input priority logic)
            influence_score = base_influence / (axiom.priority + 1) # Add 1 to avoid division by zero and give lowest priority non-zero influence
            weighted_influence[axiom.id] = influence_score

        # Normalize influences to sum to 1, if desired for a distribution
        total_normalized_influence = sum(weighted_influence.values())
        if total_normalized_influence > 0:
            for axiom_id in weighted_influence:
                weighted_influence[axiom_id] /= total_normalized_influence

        logger.debug(f"Applied axiom weighting: {weighted_influence}")
        return weighted_influence


    def generate_rules(self, axiom_set: AxiomSet, world_name: str) -> GeneratedWorldRules:
        """
        Generates a set of axiomatically aligned rules for a new simulated world.

        Args:
            axiom_set (AxiomSet): The parsed axioms guiding world generation.
            world_name (str): The name for the new world.

        Returns:
            GeneratedWorldRules: A dataclass containing the generated rules and metadata.
        """
        if not self.generative_model:
            logger.error("Generative AI model not loaded. Cannot generate rules.")
            raise RuntimeError("Generative AI model not loaded.")

        logger.info(f"Initiating rule generation for world: {world_name}")
        
        axiom_embeddings = [np.array(a.embedding) for a in axiom_set.axioms if a.embedding]
        axiom_ids = [a.id for a in axiom_set.axioms]
        axiom_influence_weights = self._apply_axiom_weighting(axiom_set.axioms)

        # --- Placeholder for complex GraphTransformer inference logic ---
        # In a real implementation, the GraphTransformer would take axiom_embeddings
        # and potentially initial graph seeds (like "Agent", "Resource") as input.
        # It would then iteratively predict new nodes (rules, entities) and edges
        # (relationships) to form the rule_graph, conditioned on axiom_influence_weights.
        # This is where the core creativity and axiom alignment is woven into the world's fabric.
        
        # For this skeleton, we'll simulate a plausible output.
        simulated_rules, simulated_rule_graph = self._simulate_rule_generation_logic(
            axiom_embeddings, axiom_ids, axiom_influence_weights, world_name
        )
        # --- End Placeholder ---

        logger.info(f"Generated {len(simulated_rules)} rules for world: {world_name}")

        world_rules = GeneratedWorldRules(
            world_name=world_name,
            rules=simulated_rules,
            rule_graph=simulated_rule_graph,
            creation_timestamp=datetime.datetime.now().isoformat(),
            axioms_used_ids=axiom_ids,
            meta_data={
                "rule_generator_model_type": self.model_config['rule_generator_model']['type'],
                "axiom_influence_distribution": axiom_influence_weights,
                "generation_parameters": self.model_config['rule_generator_model']['output_constraints']
            }
        )
        logger.success(f"Successfully compiled GeneratedWorldRules for {world_name}.")
        return world_rules

    def _simulate_rule_generation_logic(self, axiom_embeddings: List[np.ndarray], 
                                         axiom_ids: List[str], 
                                         axiom_influence_weights: Dict[str, float],
                                         world_name: str) -> (List[GeneratedRule], nx.DiGraph):
        """
        A mock function to simulate the complex GraphTransformer inference process.
        This generates plausible rules and a simple graph structure for demonstration.
        """
        logger.warning("Using simulated rule generation logic. Replace with actual GraphTransformer inference.")

        rules: List[GeneratedRule] = []
        rule_graph = nx.DiGraph()

        # Create some base entities/rules that are always present
        base_entities = [
            ("World_Root", "Represents the entire simulation container.", "meta_entity"),
            ("Agent_Spawner", "Spawns initial agents according to population rules.", "system_mechanic"),
            ("Resource_Source", "Provides renewable resources to the environment.", "environmental_mechanic"),
        ]
        
        for entity_id, desc, etype in base_entities:
            rule_graph.add_node(entity_id, type=etype, description=desc)
            rules.append(GeneratedRule(
                id=entity_id, description=desc, type=etype, parameters={}, dependencies=[],
                axiom_influence={'EPISTEMIC_COHERENCE_001': 1.0} # Base rules are axiom-driven
            ))

        # Simulate generating rules influenced by axioms
        for i, axiom_id in enumerate(axiom_ids):
            influence = axiom_influence_weights.get(axiom_id, 0.0)
            if influence < 0.1: continue # Skip very low influence axioms for simplicity

            # Generate a few rules related to this axiom
            if "FLOURISHING" in axiom_id:
                rule_id = f"Agent_Cooperation_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id,
                    description=f"Agents gain well-being from cooperative actions. Influenced by {axiom_id}.",
                    type="agent_behavior",
                    parameters={"cooperation_reward_factor": round(0.5 + influence*0.5, 2)},
                    dependencies=["Agent_Spawner"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id, type="agent_behavior")
                rule_graph.add_edge("Agent_Spawner", rule_id, relation="influences")
                
                rule_id_2 = f"Flourishing_Feedback_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id_2,
                    description=f"System monitors average agent well-being and adjusts resource generation. Strongly influenced by {axiom_id}.",
                    type="system_mechanic",
                    parameters={"resource_adjustment_sensitivity": round(0.1 + influence*0.1, 2)},
                    dependencies=["Resource_Source"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id_2, type="system_mechanic")
                rule_graph.add_edge(rule_id, rule_id_2, relation="monitors")
                rule_graph.add_edge(rule_id_2, "Resource_Source", relation="adjusts")


            elif "SUSTAINABILITY" in axiom_id:
                rule_id = f"Resource_Regen_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id,
                    description=f"Resource regeneration rate is tied to current ecological health. Influenced by {axiom_id}.",
                    type="environmental_law",
                    parameters={"regen_health_multiplier": round(0.8 + influence*0.2, 2)},
                    dependencies=["Resource_Source"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id, type="environmental_law")
                rule_graph.add_edge("Resource_Source", rule_id, relation="governed_by")
                
                rule_id_2 = f"Zero_Waste_Cycle_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id_2,
                    description=f"Consumed resources are partially returned to environment as compost. Strongly influenced by {axiom_id}.",
                    type="environmental_mechanic",
                    parameters={"waste_to_regen_ratio": round(0.1 + influence*0.1, 2)},
                    dependencies=["Resource_Source"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id_2, type="environmental_mechanic")
                rule_graph.add_edge(rule_id_2, "Resource_Source", relation="feeds")
                rule_graph.add_edge(rule_id, rule_id_2, relation="enforces")


            elif "EQUITY" in axiom_id:
                rule_id = f"Resource_Sharing_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id,
                    description=f"Agents in high-resource areas automatically share surplus with low-resource agents. Influenced by {axiom_id}.",
                    type="social_dynamic",
                    parameters={"sharing_threshold": round(0.7 - influence*0.2, 2)},
                    dependencies=["Agent_Spawner"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id, type="social_dynamic")
                rule_graph.add_edge("Agent_Spawner", rule_id, relation="applies_to")
                
                rule_id_2 = f"Opportunity_Access_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id_2,
                    description=f"Agent group with lowest average opportunity gains bonus action points. Strongly influenced by {axiom_id}.",
                    type="social_dynamic",
                    parameters={"opportunity_bonus_factor": round(0.1 + influence*0.1, 2)},
                    dependencies=["Agent_Spawner"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id_2, type="social_dynamic")
                rule_graph.add_edge(rule_id, rule_id_2, relation="influences")

            elif "COHERENCE" in axiom_id:
                rule_id = f"Rule_Consistency_Check_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id,
                    description=f"Meta-rule: All generated rules are checked for internal logical consistency before application. Influenced by {axiom_id}.",
                    type="meta_rule",
                    parameters={"consistency_tolerance": round(0.999 + influence*0.001, 3)},
                    dependencies=["World_Root"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id, type="meta_rule")
                rule_graph.add_edge("World_Root", rule_id, relation="governs")

            elif "RESILIENCE" in axiom_id:
                rule_id = f"Adaptive_Mutation_Rate_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id,
                    description=f"Agent behaviors mutate faster under environmental stress. Influenced by {axiom_id}.",
                    type="agent_behavior",
                    parameters={"stress_mutation_multiplier": round(1.0 + influence*0.5, 2)},
                    dependencies=["Agent_Spawner"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id, type="agent_behavior")
                rule_graph.add_edge("Agent_Spawner", rule_id, relation="adapts_to")

            elif "AGENCY" in axiom_id:
                rule_id = f"Agent_Autonomy_Guard_{world_name}_{i}"
                rules.append(GeneratedRule(
                    id=rule_id,
                    description=f"Agents have protected decision-making processes, even if suboptimal. Influenced by {axiom_id}.",
                    type="agent_behavior",
                    parameters={"autonomy_priority_level": round(0.8 + influence*0.2, 2)},
                    dependencies=["Agent_Spawner"],
                    axiom_influence={axiom_id: influence}
                ))
                rule_graph.add_node(rule_id, type="agent_behavior")
                rule_graph.add_edge("Agent_Spawner", rule_id, relation="grants")
            
        # Add some random inter-rule dependencies to make the graph more complex
        all_rule_ids = [r.id for r in rules]
        for _ in range(len(rules) * 2): # Add double the number of rules as random edges
            if len(all_rule_ids) < 2: break
            source, target = random.sample(all_rule_ids, 2)
            if not rule_graph.has_edge(source, target):
                rule_graph.add_edge(source, target, relation="random_influence")

        logger.debug(f"Simulated {len(rules)} rules and {rule_graph.number_of_edges()} edges.")
        return rules, rule_graph

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
                'type': "GraphTransformer",
                'architecture': {}, # Empty for placeholder
                'hyperparameters': {}, # Empty for placeholder
                'input_processing': {
                    'axiom_embedding_model': "sentence-transformers/all-MiniLM-L6-v2",
                    'max_axiom_tokens': 128,
                    'axiom_weighting_strategy': "priority_weighted_attention"
                },
                'output_constraints': {
                    'max_rule_complexity_score': 0.8
                }
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
                'initial_world_size': {'x_dim': 10, 'y_dim': 10},
                'max_simulation_steps': 1000
            }
        }
        with open(sim_settings_path, 'w') as f:
            yaml.safe_dump(dummy_sim_settings, f)
        logger.info(f"Created dummy {sim_settings_path} for testing.")

    # Create a dummy axioms.yaml if not present (from axiom_parser step)
    axiom_file_path = "config/axioms.yaml"
    if not os.path.exists(axiom_file_path):
        dummy_axioms = {
            'world_axioms': [
                {'id': 'PHILOSOPHY_FLOURISHING_001', 'principle': 'Maximize well-being.', 'priority': 1, 'type': 'ethical'},
                {'id': 'ECOLOGY_SUSTAINABILITY_001', 'principle': 'Ensure sustainability.', 'priority': 2, 'type': 'environmental'},
                {'id': 'EPISTEMIC_COHERENCE_001', 'principle': 'Maintain logical consistency.', 'priority': 0, 'type': 'foundational'},
                {'id': 'SOCIAL_EQUITY_001', 'principle': 'Minimize disparities.', 'priority': 3, 'type': 'social'},
                {'id': 'SYSTEMS_RESILIENCE_001', 'principle': 'Foster adaptability.', 'priority': 4, 'type': 'systemic'},
                {'id': 'ETHICS_AGENCY_001', 'principle': 'Protect agent autonomy.', 'priority': 1, 'type': 'ethical'},
            ]
        }
        with open(axiom_file_path, 'w') as f:
            yaml.safe_dump(dummy_axioms, f)
        logger.info(f"Created dummy {axiom_file_path} for testing.")
    
    # Also ensure src/utils/logger.py exists for setup_logging
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        # Assuming logger.py is already there from axiom_parser.py's __main__ block

    from src.core.axiom_parser import AxiomParser # Import here to ensure logger setup happens first

    # 1. Parse Axioms
    axiom_parser = AxiomParser(model_config_path=model_config_path)
    axiom_set = axiom_parser.parse_axioms(axiom_file_path)

    # 2. Generate Rules
    rule_generator = RuleGenerator(model_config_path=model_config_path, 
                                   simulation_settings_path=sim_settings_path)
    world_rules = rule_generator.generate_rules(axiom_set, "MyFirstOntologicalWorld")

    # 3. Print Results (simplified)
    logger.info(f"\n--- Generated World: {world_rules.world_name} ---")
    logger.info(f"Creation Timestamp: {world_rules.creation_timestamp}")
    logger.info(f"Axioms Used: {world_rules.axioms_used_ids}")
    logger.info(f"Total Rules Generated: {len(world_rules.rules)}")
    
    logger.info("\n--- Sample Rules ---")
    for i, rule in enumerate(world_rules.rules[:5]): # Print first 5 rules
        logger.info(f"Rule ID: {rule.id}, Type: {rule.type}")
        logger.info(f"  Desc: {rule.description[:70]}...")
        logger.info(f"  Params: {rule.parameters}")
        logger.info(f"  Axiom Influence: {rule.axiom_influence}")
        if rule.dependencies:
            logger.info(f"  Dependencies: {rule.dependencies}")
    
    logger.info(f"\n--- Rule Graph Stats ---")
    logger.info(f"Nodes in graph: {world_rules.rule_graph.number_of_nodes()}")
    logger.info(f"Edges in graph: {world_rules.rule_graph.number_of_edges()}")
    
    # You could also save the graph to a GML or JSON for later visualization
    # nx.write_gml(world_rules.rule_graph, "my_first_ontological_world_rules.gml")
    # logger.info("Rule graph saved to my_first_ontological_world_rules.gml")
