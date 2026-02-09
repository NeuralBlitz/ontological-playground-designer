# Ontological Playground Designer

## Overview
An AI-powered system for designing and evaluating axiomatically aligned simulated worlds. The project provides both a CLI and a FastAPI-based REST API for generating worlds from foundational axiomatic principles, evaluating their alignment, and detecting paradoxes.

## Recent Changes
- 2026-02-09: Initial Replit setup - fixed code bugs, installed dependencies, configured API server on port 5000

## Project Architecture

### Structure
```
├── config/                     # YAML configuration files
│   ├── axioms.yaml            # Axiom definitions (ethical principles)
│   ├── model_config.yaml      # AI model configurations
│   └── simulation_settings.yaml # Simulation parameters
├── src/
│   ├── core/                  # Core AI components
│   │   ├── axiom_parser.py    # Parses axioms, generates embeddings
│   │   ├── rule_generator.py  # Generates world rules from axioms
│   │   ├── world_compiler.py  # Compiles rules into simulation config
│   │   ├── flourishing_evaluator.py # Evaluates world alignment
│   │   └── paradox_detector.py # Detects logical/ethical contradictions
│   ├── interfaces/
│   │   ├── api.py             # FastAPI REST API (port 5000)
│   │   └── cli.py             # Typer CLI interface
│   ├── utils/
│   │   ├── logger.py          # Loguru-based logging setup
│   │   ├── graph_utils.py     # Graph utility functions
│   │   └── math_utils.py      # Math utility functions
│   └── main.py                # CLI entry point
├── data/                      # Generated outputs
│   ├── generated_worlds/      # World configuration files
│   └── evaluation_reports/    # Evaluation report files
├── models/                    # ML model storage (git-kept)
├── scripts/                   # Training and evaluation scripts
└── tests/                     # Unit tests
```

### Key Technologies
- Python 3.11
- FastAPI + Uvicorn (REST API)
- PyTorch (CPU) + HuggingFace Transformers + Sentence-Transformers
- NetworkX (graph structures)
- Loguru (logging)
- Typer (CLI)
- NumPy, SciPy, scikit-learn

### API Endpoints
- `GET /health` - Health check
- `POST /generate_world` - Generate a new world from axioms
- `POST /evaluate_world` - Evaluate a world for alignment
- `GET /docs` - Swagger UI documentation

### Running
The API server runs on port 5000 via uvicorn with auto-reload enabled.
