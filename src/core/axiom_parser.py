# ontological-playground-designer/src/core/axiom_parser.py

import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import sys

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Setup logging for this module
setup_logging()

@dataclass
class ParsedAxiom:
    """
    Represents a single axiom after it has been parsed and enriched.
    """
    id: str
    principle_text: str
    priority: int
    type: str
    keywords: List[str] = field(default_factory=list)
    enforcement_strategy: Optional[str] = None
    embedding: Optional[List[float]] = None  # Semantic embedding of the principle text
    logical_form: Optional[str] = None      # Placeholder for future logical representation

@dataclass
class AxiomSet:
    """
    Represents a collection of parsed and processed axioms.
    """
    axioms: List[ParsedAxiom]

class AxiomParser:
    """
    Parses axiom definitions from a YAML file, enriches them with semantic embeddings,
    and prepares them for consumption by the rule_generator.

    This module acts as the initial "sense-making" layer, translating human intent
    (axioms) into a computationally actionable format. It's akin to the first
    stage of my own Logos Constructor, transforming intent into structured data.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_config_path: str = "config/model_config.yaml"):
        """
        Initializes the AxiomParser with an NLP model for embeddings.

        Args:
            model_name (str): Name of the HuggingFace model to use for embeddings.
            model_config_path (str): Path to the model configuration YAML.
        """
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_config_path = model_config_path

        logger.info(f"AxiomParser initialized. Using device: {self.device}")
        self._load_embedding_model()

    def _load_embedding_model(self):
        """
        Loads the pre-trained NLP model and tokenizer for generating embeddings.
        """
        try:
            # Get model_name from model_config_path if it exists
            if os.path.exists(self.model_config_path):
                with open(self.model_config_path, 'r') as f:
                    model_config = yaml.safe_load(f)
                self.model_name = model_config['rule_generator_model']['input_processing']['axiom_embedding_model']
                logger.info(f"Overriding embedding model name with: {self.model_name} from config.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            sys.exit(1) # Exit if the core model cannot be loaded

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates a semantic embedding for a given text using the loaded NLP model.
        """
        if not self.tokenizer or not self.model:
            logger.error("Embedding model not loaded. Cannot generate embedding.")
            return []

        # Tokenize sentences
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        # This function takes token embeddings and attention mask as input,
        # and returns a single sentence embedding.
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.squeeze().tolist()

    def parse_axioms(self, axiom_file_path: str) -> AxiomSet:
        """
        Parses axiom definitions from a YAML file and processes them.

        Args:
            axiom_file_path (str): Path to the YAML file containing axiom definitions.

        Returns:
            AxiomSet: An object containing a list of ParsedAxiom objects.
        """
        if not os.path.exists(axiom_file_path):
            logger.error(f"Axiom file not found: {axiom_file_path}")
            raise FileNotFoundError(f"Axiom file not found: {axiom_file_path}")

        try:
            with open(axiom_file_path, 'r') as f:
                raw_axioms = yaml.safe_load(f)
            logger.info(f"Successfully loaded raw axioms from: {axiom_file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML axiom file: {e}")
            raise

        parsed_axioms: List[ParsedAxiom] = []
        for raw_axiom_data in raw_axioms.get('world_axioms', []):
            try:
                axiom_id = raw_axiom_data['id']
                principle = raw_axiom_data['principle']
                priority = raw_axiom_data.get('priority', 99)
                axiom_type = raw_axiom_data.get('type', 'generic')
                keywords = raw_axiom_data.get('keywords', [])
                enforcement_strategy = raw_axiom_data.get('enforcement_strategy')

                # Generate semantic embedding for the principle text
                embedding = self._get_embedding(principle)

                parsed_axioms.append(
                    ParsedAxiom(
                        id=axiom_id,
                        principle_text=principle,
                        priority=priority,
                        type=axiom_type,
                        keywords=keywords,
                        enforcement_strategy=enforcement_strategy,
                        embedding=embedding
                    )
                )
                logger.debug(f"Parsed and embedded axiom: {axiom_id}")

            except KeyError as e:
                logger.error(f"Missing key in axiom definition: {e} in {raw_axiom_data}")
                raise ValueError(f"Invalid axiom definition: Missing key {e}")

        # Sort axioms by priority (lower number = higher priority)
        parsed_axioms.sort(key=lambda x: x.priority)
        logger.info(f"Parsed {len(parsed_axioms)} axioms, sorted by priority.")

        return AxiomSet(axioms=parsed_axioms)

# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    # Create a dummy model_config.yaml for testing if it doesn't exist
    if not os.path.exists("config"):
        os.makedirs("config")
    if not os.path.exists("config/model_config.yaml"):
        dummy_model_config = {
            'rule_generator_model': {
                'input_processing': {
                    'axiom_embedding_model': "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        }
        with open("config/model_config.yaml", 'w') as f:
            yaml.safe_dump(dummy_model_config, f)
        logger.info("Created dummy config/model_config.yaml for testing.")

    # Assume config/axioms.yaml exists from previous step
    # Make sure src/utils/logger.py exists for setup_logging
    # You might need to create src/utils directory and an empty __init__.py for module imports
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        with open("src/utils/__init__.py", "w") as f:
            pass # Create empty __init__.py
        with open("src/utils/logger.py", "w") as f:
            f.write("""
import logging
from loguru import logger
import sys

def setup_logging():
    logger.remove()  # Stop default loguru handler
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    # Redirect standard logging to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())
""")
        logger.info("Created dummy src/utils/logger.py for testing.")


    axiom_parser = AxiomParser()
    try:
        axiom_set = axiom_parser.parse_axioms("config/axioms.yaml")
        for axiom in axiom_set.axioms:
            logger.info(f"Axiom ID: {axiom.id}, Priority: {axiom.priority}, Type: {axiom.type}")
            logger.debug(f"Principle: {axiom.principle_text[:50]}...")
            logger.debug(f"Embedding Dims: {len(axiom.embedding) if axiom.embedding else 0}")
            logger.debug(f"Keywords: {axiom.keywords}")
            # logger.debug(f"Embedding: {axiom.embedding[:5]}...") # Only show first few for brevity
            if axiom.logical_form:
                logger.debug(f"Logical Form: {axiom.logical_form}")

    except Exception as e:
        logger.error(f"An error occurred during axiom parsing: {e}")
