# ontological-playground-designer/tests/unit/test_axiom_parser.py

import pytest
import os
import yaml
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.axiom_parser import AxiomParser, AxiomSet, ParsedAxiom
from src.utils.logger import setup_logging # Ensure logging is set up for tests
from src.config.model_config import rule_generator_model # Assuming this exists or mocked later

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_axiom_parser.log")

# --- Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def dummy_model_config_path(tmp_path_factory):
    """Creates a temporary model_config.yaml for testing."""
    config_dir = tmp_path_factory.mktemp("config")
    path = config_dir / "model_config.yaml"
    dummy_config = {
        'rule_generator_model': {
            'input_processing': {
                'axiom_embedding_model': "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    }
    with open(path, 'w') as f:
        yaml.safe_dump(dummy_config, f)
    return str(path)

@pytest.fixture(scope="module")
def mock_embedding_model():
    """Mocks the AutoTokenizer and AutoModel for _get_embedding."""
    with patch('src.core.axiom_parser.AutoTokenizer.from_pretrained') as mock_tokenizer_load, \
         patch('src.core.axiom_parser.AutoModel.from_pretrained') as mock_model_load:
        
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Configure mock_tokenizer to return a dummy encoded_input
        mock_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.side_effect = lambda *args, **kwargs: MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]), 
            attention_mask=torch.tensor([[1, 1, 1]]),
            to=lambda x: MagicMock(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]]))
        )

        # Configure mock_model to return a dummy model_output
        mock_model.return_value = mock_model
        mock_model.side_effect = lambda *args, **kwargs: MagicMock(
            last_hidden_state=torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]), # token embeddings
            pooler_output=torch.tensor([[0.7, 0.8]]), # pooled output if model has it
            __getitem__=lambda x: torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]) # For model_output[0] access
        )
        
        # Patch F.normalize as well, so it just returns a dummy normalized tensor
        with patch('src.core.axiom_parser.F.normalize') as mock_normalize:
            mock_normalize.return_value = torch.tensor([[0.5, 0.5]]) # Example normalized embedding
            yield mock_tokenizer_load, mock_model_load # Yield control to tests
            # Cleanup happens after tests complete

@pytest.fixture
def parser_instance(dummy_model_config_path, mock_embedding_model):
    """Returns an AxiomParser instance with mocked dependencies."""
    # The mocks are active due to mock_embedding_model fixture
    return AxiomParser(model_config_path=dummy_model_config_path)

@pytest.fixture
def valid_axiom_file(tmp_path):
    """Creates a temporary valid axioms.yaml file."""
    path = tmp_path / "valid_axioms.yaml"
    content = """
world_axioms:
  - id: PHILOSOPHY_FLOURISHING_001
    principle: "Maximize the long-term well-being and adaptive capacity of all sentient agents."
    priority: 1
    type: ethical
    keywords: ["well-being", "flourishing"]
    enforcement_strategy: "Design intrinsic feedback loops."
  - id: ECOLOGY_SUSTAINABILITY_001
    principle: "Ensure all resource consumption rates are perpetually sustainable."
    priority: 2
    type: environmental
    keywords: ["sustainability", "resources"]
  - id: EPISTEMIC_COHERENCE_001
    principle: "Maintain absolute logical and conceptual coherence."
    priority: 0
    type: foundational
    keywords: ["coherence", "logic"]
    """
    with open(path, 'w') as f:
        f.write(content)
    return str(path)

@pytest.fixture
def invalid_axiom_file_missing_key(tmp_path):
    """Creates a temporary invalid axioms.yaml file (missing 'principle')."""
    path = tmp_path / "invalid_axioms_missing_key.yaml"
    content = """
world_axioms:
  - id: INVALID_AXIOM_001
    priority: 1 # Missing principle
    type: ethical
    """
    with open(path, 'w') as f:
        f.write(content)
    return str(path)

@pytest.fixture
def empty_axiom_file(tmp_path):
    """Creates a temporary empty axioms.yaml file."""
    path = tmp_path / "empty_axioms.yaml"
    content = """
world_axioms: []
    """
    with open(path, 'w') as f:
        f.write(content)
    return str(path)

@pytest.fixture
def axiom_file_no_world_axioms_key(tmp_path):
    """Creates a temporary axioms.yaml file without the 'world_axioms' key."""
    path = tmp_path / "no_world_axioms_key.yaml"
    content = """
other_data: "This file has no axioms"
    """
    with open(path, 'w') as f:
        f.write(content)
    return str(path)

# --- Test Cases ---

def test_axiom_parser_initialization(parser_instance):
    """Tests if the AxiomParser initializes correctly and loads the model placeholder."""
    assert parser_instance is not None
    assert parser_instance.tokenizer is not None
    assert parser_instance.model is not None
    assert parser_instance.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("Test: AxiomParser initializes correctly.")

def test_parse_valid_axioms(parser_instance, valid_axiom_file):
    """Tests parsing of a valid axiom file."""
    axiom_set = parser_instance.parse_axioms(valid_axiom_file)
    assert isinstance(axiom_set, AxiomSet)
    assert len(axiom_set.axioms) == 3
    
    # Check sorting by priority (0 is highest)
    assert axiom_set.axioms[0].id == "EPISTEMIC_COHERENCE_001"
    assert axiom_set.axioms[1].id == "PHILOSOPHY_FLOURISHING_001"
    assert axiom_set.axioms[2].id == "ECOLOGY_SUSTAINABILITY_001"
    
    # Check content of a parsed axiom
    flourishing_axiom = axiom_set.axioms[1]
    assert flourishing_axiom.principle_text.startswith("Maximize the long-term well-being")
    assert flourishing_axiom.type == "ethical"
    assert "well-being" in flourishing_axiom.keywords
    assert flourishing_axiom.embedding is not None and len(flourishing_axiom.embedding) > 0
    logger.info("Test: Valid axioms parsed and sorted correctly.")

def test_parse_axiom_file_not_found(parser_instance):
    """Tests error handling for a non-existent axiom file."""
    with pytest.raises(FileNotFoundError):
        parser_instance.parse_axioms("non_existent_file.yaml")
    logger.info("Test: FileNotFoundError correctly raised for non-existent file.")

def test_parse_invalid_axiom_file_missing_key(parser_instance, invalid_axiom_file_missing_key):
    """Tests error handling for an axiom file with a missing required key."""
    with pytest.raises(ValueError, match="Missing key in axiom definition: 'principle'"):
        parser_instance.parse_axioms(invalid_axiom_file_missing_key)
    logger.info("Test: ValueError correctly raised for missing axiom key.")

def test_parse_empty_axiom_file(parser_instance, empty_axiom_file):
    """Tests parsing an empty axiom file (empty 'world_axioms' list)."""
    axiom_set = parser_instance.parse_axioms(empty_axiom_file)
    assert isinstance(axiom_set, AxiomSet)
    assert len(axiom_set.axioms) == 0
    logger.info("Test: Empty axiom file parsed correctly.")

def test_parse_axiom_file_no_world_axioms_key(parser_instance, axiom_file_no_world_axioms_key):
    """Tests parsing a file that lacks the 'world_axioms' top-level key."""
    axiom_set = parser_instance.parse_axioms(axiom_file_no_world_axioms_key)
    assert isinstance(axiom_set, AxiomSet)
    assert len(axiom_set.axioms) == 0
    logger.info("Test: File without 'world_axioms' key parsed correctly (empty axiom set).")

@patch('src.core.axiom_parser.torch.cuda.is_available', return_value=True)
def test_device_selection_cuda_available(mock_cuda_available, parser_instance):
    """Tests if CUDA device is selected when available."""
    assert parser_instance.device == torch.device("cuda")
    logger.info("Test: CUDA device selected when available.")

@patch('src.core.axiom_parser.torch.cuda.is_available', return_value=False)
def test_device_selection_cuda_not_available(mock_cuda_available, parser_instance):
    """Tests if CPU device is selected when CUDA is not available."""
    assert parser_instance.device == torch.device("cpu")
    logger.info("Test: CPU device selected when CUDA is not available.")

# You can add more complex tests for embedding consistency if you have known embeddings for specific texts
# However, mocking the model output is generally sufficient for unit testing the parser's logic.
