# ontological-playground-designer/src/utils/logger.py

import logging
from loguru import logger
import sys
import os

# --- Logger Configuration ---
# This module centralizes logging for the entire Ontological Playground Designer.
# It uses loguru for structured, colorized, and configurable logging,
# enhancing traceability and debugging.

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configures loguru to provide structured and consistent logging across the project.
    It removes default handlers and adds a custom one for console output and optionally to a file.

    Args:
        log_level (str): The minimum logging level to display (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        log_file (Optional[str]): Path to a file where logs should also be written.
                                   If None, logs only to console.
    """
    logger.remove()  # Remove default handler to control logging entirely

    # Add handler for console output (stderr for errors/warnings, stdout for info/debug)
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        # Format includes time, level, module:function:line, and message
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        filter=lambda record: record["level"].name in ["INFO", "DEBUG", "WARNING", "ERROR", "SUCCESS"]
    )
    
    # Add a custom SUCCESS level, as loguru doesn't have it by default
    logger.level("SUCCESS", no=25, color="<green>", icon="✅")

    # Optionally add a file handler
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            logger.add(
                log_file,
                level=log_level,
                rotation="10 MB",  # Rotate log file if it exceeds 10 MB
                compression="zip",  # Compress old log files
                enqueue=True,       # Use a queue for writing logs to file for non-blocking I/O
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
            )
            logger.info(f"Logging configured to file: {log_file} with level: {log_level}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}")

    # Redirect standard Python logging messages to Loguru
    # This is important for libraries that use standard logging module
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    logger.debug("Standard logging redirected to Loguru.")


class InterceptHandler(logging.Handler):
    """
    Intercepts standard Python logging messages and redirects them to Loguru.
    This ensures all log messages from any library are handled consistently.
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where loguru message originated, so it's not always this interceptor
        frame, depth = logging.currentframe(), 0
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    # Ensure src/utils directory exists
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        
    log_output_file = "logs/test_ontology_designer.log"
    setup_logging(log_level="DEBUG", log_file=log_output_file)

    logger.debug("This is a DEBUG message from the main script.")
    logger.info("This is an INFO message, indicating normal operation.")
    logger.success("This is a SUCCESS message, indicating a successful task completion.")
    logger.warning("This is a WARNING message, something might be off.")
    
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("An ERROR occurred: Division by zero!", exc_info=True) # exc_info=True logs traceback

    # Test standard logging redirect
    std_logger = logging.getLogger("StandardLibTest")
    std_logger.info("This message from standard logging should appear in Loguru output.")
    std_logger.warning("Standard warning detected.")
    
    logger.info(f"Log messages also written to: {log_output_file}")
