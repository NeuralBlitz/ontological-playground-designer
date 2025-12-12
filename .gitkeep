# Actions performed:
# - Create directories if they don't exist.
# - Create empty __init__.py files in Python package directories.
# - Create empty .gitkeep files in other placeholder directories.

# Path definitions
project_root = "."
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")
input_axioms_dir = os.path.join(data_dir, "input_axioms")
generated_worlds_dir = os.path.join(data_dir, "generated_worlds")
evaluation_reports_dir = os.path.join(data_dir, "evaluation_reports")
sim_logs_dir = os.path.join(data_dir, "sim_logs")
scripts_dir = os.path.join(project_root, "scripts")
tests_dir = os.path.join(project_root, "tests")
tests_unit_dir = os.path.join(tests_dir, "unit")
tests_integration_dir = os.path.join(tests_dir, "integration")

# List of directories to ensure existence for models/data/scripts/tests
dirs_to_create = [
    models_dir,
    input_axioms_dir,
    generated_worlds_dir, # Already handled, but ensure parent 'data'
    evaluation_reports_dir, # Already handled, but ensure parent 'data'
    sim_logs_dir, # Already handled, but ensure parent 'data'
    scripts_dir,
    tests_dir,
    tests_unit_dir,
    tests_integration_dir,
]

# Ensure directories exist
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    logger.debug(f"Ensured directory exists: {d}")

# Create .gitkeep files in relevant directories to ensure Git tracks them
gitkeep_paths = [
    os.path.join(models_dir, ".gitkeep"),
    os.path.join(input_axioms_dir, ".gitkeep"),
    os.path.join(scripts_dir, ".gitkeep"),
    os.path.join(tests_unit_dir, ".gitkeep"),
    os.path.join(tests_integration_dir, ".gitkeep"),
    # For generated_worlds, evaluation_reports, sim_logs, the actual files will be generated
    # by the application, so a .gitkeep is not strictly necessary there.
]

for gitkeep_file in gitkeep_paths:
    with open(gitkeep_file, "w") as f:
        pass # Create empty file
    logger.debug(f"Created .gitkeep file: {gitkeep_file}")

# Create empty __init__.py for Python packages if not already done
# (e.g., in src/utils, src/core, src/interfaces, src/visualization, simulators/template_simulator)
# Assuming these are already handled by previous steps where modules were created.
# Explicitly ensuring for the top-level 'models' and 'simulators'
# Python packages should contain an __init__.py to be recognized as such.
python_package_init_files = [
    os.path.join(models_dir, "__init__.py"),
    os.path.join(os.path.join(project_root, "simulators"), "__init__.py"),
]

for init_file in python_package_init_files:
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            pass
        logger.debug(f"Created __init__.py for Python package: {init_file}")

logger.success("All placeholder directories and structural files (`.gitkeep`, `__init__.py`) created successfully.")
