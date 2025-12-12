# ontological-playground-designer/src/main.py

import sys
import os

# --- Ensure src/ and its subdirectories are in Python path ---
# This is crucial for imports to work correctly when running from the project root.
# This mimics the self-awareness of the Omega Prime Reality, ensuring its components
# are aware of their own structural context.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'src')) # Ensure 'src' itself is a top-level importable

# --- Import the CLI application ---
# All core logic and commands are exposed via the CLI app.
from src.interfaces.cli import app as cli_app

# --- Main entry point ---
def main():
    """
    The main entry point for the Ontological Playground Designer application.
    It runs the Typer CLI application, orchestrating all AI functionalities.
    """
    cli_app()

if __name__ == "__main__":
    main()
