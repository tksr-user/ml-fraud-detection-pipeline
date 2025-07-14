import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta

# === Load Env Variables ===
load_dotenv()
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")

# === Arize Info ===
MODEL_ID = "fraud-model"
VERSIONS = ["v1", "v2"]
ENVIRONMENT = "PRODUCTION"

# === Date Range ===
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=90)

# === Arize REST API Headers ===
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ARIZE_API_KEY}"
}

# === Fetch metrics for each version ===
model_metrics = {}
for version in VERSIONS:
    url = f"https://api.arize.com/v1/performance"
    payload = {
        "space_key": ARIZE_SPACE_ID,
        "model_id": MODEL_ID,
        "model_version": version,
        "environment": ENVIRONMENT,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat()
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f" Failed to fetch metrics for version {version}. Status Code: {response.status_code}")
        continue

    data = response.json().get("data", {})
    metrics = data.get("performance_metrics", {})

    model_metrics[version] = {
        "accuracy": metrics.get("accuracy"),
        "f1_score": metrics.get("f1_score"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "drift_score": metrics.get("prediction_drift_score")
    }

# === Print All Metrics ===
print("\n Model Metrics Comparison (Last 90 Days):")
for version, metrics in model_metrics.items():
    print(f"\n🔹 Version {version}")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# === Build Prompt for Ollama ===
prompt = f"""
Compare these two versions of a fraud detection model:

Version v1:
{model_metrics.get('v1')}

Version v2:
{model_metrics.get('v2')}

Which model should be used in production and why?
Please analyze based on: accuracy, f1_score, and drift_score.
Keep the explanation clear and decision-focused.
"""

# === Call Ollama Locally ===
print("\n Asking Ollama AI Agent to analyze...")
ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",  # or use "mistral"
        "prompt": prompt,
        "stream": False
    }
)

ollama_output = ollama_response.json().get("response", "No response from Ollama")
print("\n Ollama Recommendation:\n")
print(ollama_output)