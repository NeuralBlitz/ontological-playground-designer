# ontological-playground-designer/tests/unit/test_logger.py

import pytest
import os
import sys
import logging
from loguru import logger
from unittest.mock import patch, MagicMock

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import setup_logging and InterceptHandler directly
from src.utils.logger import setup_logging, InterceptHandler

# --- Fixtures for reusable test data ---

@pytest.fixture(autouse=True)
def clean_loguru_handlers():
    """Ensures loguru handlers are clean before and after each test."""
    original_handlers = list(logger._core.handlers.keys())
    yield
    # Remove any handlers added during the test
    for handler_id in list(logger._core.handlers.keys()):
        if handler_id not in original_handlers:
            logger.remove(handler_id)
    logger.remove() # Ensure all are removed to start fresh
    logger.add(sys.stderr) # Add back a default handler for normal console output after tests

@pytest.fixture
def mock_add_logger():
    """Mocks loguru.logger.add to capture calls."""
    with patch('src.utils.logger.logger.add') as mock_add:
        yield mock_add

@pytest.fixture
def mock_logger_remove():
    """Mocks loguru.logger.remove to capture calls."""
    with patch('src.utils.logger.logger.remove') as mock_remove:
        yield mock_remove

@pytest.fixture
def mock_logger_level():
    """Mocks loguru.logger.level to capture calls for custom levels."""
    with patch('src.utils.logger.logger.level') as mock_level:
        mock_level.return_value = MagicMock(name="SUCCESS") # Default return for SUCCESS
        yield mock_level

@pytest.fixture
def mock_logging_basic_config():
    """Mocks logging.basicConfig to capture calls."""
    with patch('src.utils.logger.logging.basicConfig') as mock_basic_config:
        yield mock_basic_config

@pytest.fixture
def mock_logger_opt_log():
    """Mocks logger.opt().log to capture log messages redirected from stdlib."""
    with patch('src.utils.logger.logger.opt') as mock_opt:
        mock_log_method = MagicMock()
        mock_opt.return_value.log = mock_log_method
        yield mock_log_method

# --- Test Cases for setup_logging ---

def test_setup_logging_removes_existing_handlers(mock_logger_remove, mock_add_logger, mock_logger_level, mock_logging_basic_config):
    """Verifies setup_logging calls logger.remove() at the start."""
    setup_logging()
    mock_logger_remove.assert_called_once()
    logger.info("Test: setup_logging removes existing handlers.")

def test_setup_logging_adds_console_handler(mock_add_logger, mock_logger_remove, mock_logger_level, mock_logging_basic_config):
    """Verifies setup_logging adds at least one console handler."""
    setup_logging(log_level="INFO")
    # Check that logger.add was called at least once
    assert mock_add_logger.called
    # Check if a handler for sys.stderr was added
    call_args = mock_add_logger.call_args_list
    assert any(sys.stderr == args[0] for args, kwargs in call_args)
    assert any(kwargs.get('level') == "INFO" for args, kwargs in call_args)
    logger.info("Test: setup_logging adds console handler with correct level.")

def test_setup_logging_adds_file_handler(mock_add_logger, mock_logger_remove, mock_logger_level, mock_logging_basic_config, tmp_path):
    """Verifies setup_logging adds a file handler when log_file is provided."""
    log_file = tmp_path / "test.log"
    setup_logging(log_level="DEBUG", log_file=str(log_file))
    
    # Check if a handler for the specific file was added
    call_args = mock_add_logger.call_args_list
    assert any(args[0] == str(log_file) for args, kwargs in call_args)
    assert any(kwargs.get('level') == "DEBUG" for args, kwargs in call_args)
    assert any(kwargs.get('rotation') == "10 MB" for args, kwargs in call_args)
    logger.info("Test: setup_logging adds file handler with correct parameters.")

def test_setup_logging_configures_custom_success_level(mock_logger_level, mock_add_logger, mock_logger_remove, mock_logging_basic_config):
    """Verifies setup_logging configures the custom SUCCESS level."""
    setup_logging()
    mock_logger_level.assert_any_call("SUCCESS", no=25, color="<green>", icon="✅")
    logger.info("Test: setup_logging configures custom SUCCESS level.")

def test_setup_logging_redirects_standard_logging(mock_logging_basic_config, mock_add_logger, mock_logger_remove, mock_logger_level):
    """Verifies setup_logging calls logging.basicConfig to redirect standard logging."""
    setup_logging()
    mock_logging_basic_config.assert_called_once()
    assert isinstance(mock_logging_basic_config.call_args[0][0]['handlers'][0], InterceptHandler)
    logger.info("Test: setup_logging redirects standard logging.")

# --- Test Cases for InterceptHandler ---

def test_intercept_handler_emits_to_loguru(mock_logger_opt_log, mock_add_logger, mock_logger_remove, mock_logger_level, mock_logging_basic_config):
    """Verifies that InterceptHandler correctly redirects a standard log message to loguru."""
    setup_logging() # Activate the setup with the mock add/remove
    
    # Create a standard logger
    std_logger = logging.getLogger("TestStandardLogger")
    std_logger.setLevel(logging.INFO)
    
    # Ensure InterceptHandler is active. If setup_logging is mocked, basicConfig might not run.
    # So we manually add an InterceptHandler to a test logger for this test.
    test_handler = InterceptHandler()
    std_logger.addHandler(test_handler)

    test_message = "This is a standard info message."
    std_logger.info(test_message)

    mock_logger_opt_log.assert_called_once_with("INFO", test_message)
    logger.info("Test: InterceptHandler emits standard logging to loguru.")
    
    std_logger.removeHandler(test_handler) # Clean up handler

def test_intercept_handler_captures_exception_info(mock_logger_opt_log, mock_add_logger, mock_logger_remove, mock_logger_level, mock_logging_basic_config):
    """Verifies InterceptHandler captures exception info."""
    setup_logging()
    std_logger = logging.getLogger("TestExceptionLogger")
    std_logger.setLevel(logging.ERROR)
    
    test_handler = InterceptHandler()
    std_logger.addHandler(test_handler)

    try:
        1 / 0
    except ZeroDivisionError:
        std_logger.error("Error occurred", exc_info=True)
    
    # Check if logger.opt was called with exception info
    mock_logger_opt_log.assert_called_once()
    call_args, kwargs = mock_logger_opt_log.call_args
    assert call_args[0] == "ERROR"
    assert call_args[1] == "Error occurred"
    assert kwargs['exception'] is not None
    assert isinstance(kwargs['exception'], tuple) # (type, value, traceback)
    logger.info("Test: InterceptHandler captures exception info.")

    std_logger.removeHandler(test_handler)
