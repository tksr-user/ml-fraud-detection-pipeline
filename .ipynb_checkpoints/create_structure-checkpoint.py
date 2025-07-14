import os

project_root = "."

folders = [
    "config",
    "data/raw",
    "data/processed",
    "data/results",
    "etl",
    "models",
    "experiments",
    "agent",
    "scripts",
    "utils"
]

files = [
    "config/pipeline_config.yaml",
    "etl/__init__.py",
    "etl/extract.py",
    "etl/transform.py",
    "etl/load.py",
    "models/__init__.py",
    "models/train.py",
    "models/evaluate.py",
    "models/deploy.py",
    "experiments/__init__.py",
    "experiments/run_experiment.py",
    "experiments/store_results.py",
    "experiments/compare.py",
    "agent/__init__.py",
    "agent/agent.py",
    "scripts/run_pipeline.py",
    "utils/__init__.py",
    "utils/logger.py",
    "utils/file_ops.py",
    "requirements.txt",
    "README.md"
]

for folder in folders:
    full_path = os.path.join(project_root, folder)
    os.makedirs(full_path, exist_ok=True)

for file in files:
    full_path = os.path.join(project_root, file)
    if not os.path.exists(full_path):
        with open(full_path, "w", encoding="utf-8") as f:
            if file.endswith("__init__.py"):
                f.write("# Package initializer\n")
            elif file.endswith(".py"):
                f.write(f"# {os.path.basename(file)} - Auto-generated\n")
            elif file.endswith(".yaml"):
                f.write("# Configuration settings\n")
            elif file.endswith("requirements.txt"):
                f.write("pandas\nscikit-learn\njoblib\n")
            elif file.endswith("README.md"):
                f.write("# ML Pipeline Project\n\nAutomated ETL → Train → Deploy pipeline.\n")

print("✅ Project structure created successfully!")