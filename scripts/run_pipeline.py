import sys
from etl import extract, transform, load
from models import train, deploy
from experiments import run_experiment, store_results
from utils.logger import log
import os

# Allow dataset path as argument (e.g., D1.csv)
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    dataset_path = "data/raw/creditcard.csv"

PROCESSED = "data/processed/processed_data.csv"
RESULTS = "data/results"
MODEL_PATH = "models/fraud_model.joblib"

# Step 1: Extract
df = extract.read_specific_file(dataset_path)

# Step 2: Transform
df_transformed = transform.create_features(df)

# Step 3: Load
load.save_transformed_data(df_transformed, PROCESSED)

# Step 4: Train
model = train.train_model(df_transformed)

# Step 5: Evaluate
results = run_experiment.run(model, df_transformed)
store_results.save(results, RESULTS)

# Step 6: Deploy to file
deploy.deploy_model(model, MODEL_PATH)
log("Model training & deployment complete.")