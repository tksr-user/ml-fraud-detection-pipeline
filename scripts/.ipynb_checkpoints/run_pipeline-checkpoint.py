import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etl import extract, transform, load
from models import train, deploy
from experiments import run_experiment, store_results, compare
from agent import agent


# Dynamically set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Corrected Paths
RAW = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED = os.path.join(PROJECT_ROOT, "data/processed/processed_data.csv")
NEW_EXP = os.path.join(PROJECT_ROOT, "data/results/exp1.json")
OLD_EXP = os.path.join(PROJECT_ROOT, "data/results/exp0.json")

# Pipeline
df = extract.read_and_delete_old_data(RAW)
df_transformed = transform.create_features(df)
load.write_csv(df_transformed, PROCESSED)

model = train.train_model(PROCESSED)
results = run_experiment.run(model, df_transformed)
store_results.save(results, NEW_EXP)

comparison = compare.run(NEW_EXP, OLD_EXP)
if comparison["performance_improved"]:
    deploy.deploy_model(model)

agent.start(comparison)