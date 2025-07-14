import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from arize.pandas.logger import Client
from arize.utils.types import Schema, ModelTypes, Environments

# === Load environment variables ===
load_dotenv()
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")

# === File paths ===
DATA_PATH = "data/splits/d1.csv"
MODEL_PATH = "models/fraud_model_d1.joblib"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# === Load dataset ===
df = pd.read_csv(DATA_PATH)
print(f"📄 Loaded dataset from: {DATA_PATH}")

X = df.drop(columns=["Class"])
y = df["Class"]
print(" Split dataset into features and target.")

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Set MLflow tracking URI ===
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("RandomForest_Optuna_D1")

# === Optuna objective function ===
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 4, 20)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run(nested=True):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
    
    return accuracy

# === Run Optuna Study ===
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# === Get best params and retrain ===
best_params = study.best_params
print(f" Best hyperparameters: {best_params}")

final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Final accuracy on D1: {accuracy:.4f}")

# === Save model ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
print(f" Model saved at: {MODEL_PATH}")

# === Log final model to MLflow ===
with mlflow.start_run(run_name="final_model_d1"):
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(final_model, artifact_path="model")
    print(" Final model logged to MLflow")

# === Prepare Arize logs ===
df_logs = pd.DataFrame({
    "prediction_id": [f"id_{i}" for i in range(len(y_test))],
    "prediction": y_pred,
    "actual": y_test.reset_index(drop=True),
})
print(" Prepared prediction logs for Arize")

# === Deploy to Arize ===
if not ARIZE_API_KEY or not ARIZE_SPACE_ID:
    raise EnvironmentError("Missing ARIZE_API_KEY or ARIZE_SPACE_ID in environment variables.")

client = Client(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY)

schema = Schema(
    prediction_id_column_name="prediction_id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual"
)

response = client.log(
    model_id="fraud-model",
    model_version="v1",
    model_type=ModelTypes.BINARY_CLASSIFICATION,
    environment=Environments.PRODUCTION,
    dataframe=df_logs,
    schema=schema
)

print(f" Arize Log Status: {response.status_code}")
if response.status_code == 200:
    print(" Successfully deployed D1 model to Arize.")
else:
    print(f" Deployment failed with status code: {response.status_code}")