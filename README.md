# 🛡️ Credit Card Fraud Detection - ML Pipeline

This project implements a **complete machine learning pipeline** to detect credit card fraud. It includes data preprocessing, feature storage in AWS S3, experiment tracking with MLflow, real-time model monitoring using Arize AI, and automated comparisons between MLflow and Arize logs.

---

## 📦 Project Features

- **Feature Engineering & Storage**
  - Load and process raw CSV data
  - Save features as **Parquet** files
  - Upload to **Amazon S3** for scalable access

- **Experiment Tracking**
  - Track models, parameters, and metrics using **MLflow**
  - Visualize training runs with the **MLflow UI**

- **Model Monitoring**
  - Monitor drift and performance metrics with **Arize AI**
  - Send batch prediction logs to Arize for evaluation

- **AI Agent Automation**
  - Compare logged metrics from Arize and MLflow automatically
  - Detect drift or inconsistencies using a custom agent

---

## 🗂️ Folder Structure

ml-fraud-detection-pipeline/
├── data/ # Raw or sample data
│ └── creditcard.csv
├── src/ # Source code
│ ├── data_pipeline.py # Loads data, saves Parquet, uploads to S3
│ ├── feature_store.py # (Optional) Feature store interface
│ ├── mlflow_logger.py # Logs experiments to MLflow
│ ├── arize_logger.py # Sends logs to Arize AI
│ └── ai_agent.py # Automated comparison logic
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── LICENSE # MIT License


---

 🛡️ Credit Card Fraud Detection - ML Pipeline

This project implements a **complete machine learning pipeline** to detect credit card fraud. It includes data preprocessing, feature storage in AWS S3, experiment tracking with MLflow, real-time model monitoring using Arize AI, and automated comparisons between MLflow and Arize logs.

---

## 📦 Project Features

- **Feature Engineering & Storage**
  - Load and process raw CSV data
  - Save features as **Parquet** files
  - Upload to **Amazon S3** for scalable access

- **Experiment Tracking**
  - Track models, parameters, and metrics using **MLflow**
  - Visualize training runs with the **MLflow UI**

- **Model Monitoring**
  - Monitor drift and performance metrics with **Arize AI**
  - Send batch prediction logs to Arize for evaluation

- **AI Agent Automation**
  - Compare logged metrics from Arize and MLflow automatically
  - Detect drift or inconsistencies using a custom agent

---

## 🗂️ Folder Structure

ml-fraud-detection-pipeline/
├── data/ # Raw or sample data
│ └── creditcard.csv
├── src/ # Source code
│ ├── data_pipeline.py # Loads data, saves Parquet, uploads to S3
│ ├── feature_store.py # (Optional) Feature store interface
│ ├── mlflow_logger.py # Logs experiments to MLflow
│ ├── arize_logger.py # Sends logs to Arize AI
│ └── ai_agent.py # Automated comparison logic
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── LICENSE # MIT License

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1. ✅ Clone the repository

```bash
git clone https://github.com/your-username/ml-fraud-detection-pipeline.git
cd ml-fraud-detection-pipeline
2. ✅ Install dependencies
bash
Copy code
pip install -r requirements.txt
3. ✅ Configure environment variables
You can export these in your terminal, .env file, or use os.environ in Python.

bash
Copy code
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

export ARIZE_SPACE_KEY=your-arize-space-key
export ARIZE_API_KEY=your-arize-api-key
🚀 How to Run the Pipeline
A. Run the Data Pipeline (Preprocessing + Upload to S3)
bash
Copy code
python src/data_pipeline.py
This script:

Loads creditcard.csv

Extracts features

Saves as creditcard_features.parquet

Uploads to s3://your-bucket/features/creditcard_features.parquet

B. Log to MLflow (Parameters, Metrics, Artifacts)
Start the MLflow UI in a new terminal:

bash
Copy code
mlflow ui
Then run:

bash
Copy code
python src/mlflow_logger.py
✅ View your runs at: http://127.0.0.1:5000

C. Log to Arize AI (Monitoring)
bash
Copy code
python src/arize_logger.py
Check drift/metrics at: https://app.arize.com

D. Compare Arize vs MLflow Logs (AI Agent)
bash
Copy code
python src/ai_agent.py
This script compares key metrics between MLflow and Arize to catch any drift or inconsistencies.

🧪 Example MLflow Logging Code
python
Copy code
import mlflow

with mlflow.start_run(run_name="baseline_model") as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", 0.98)
    mlflow.log_artifact("outputs/model.pkl")
🧾 Example Arize Logging Code
python
Copy code
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments

client = Client(space_key="...", api_key="...")

client.log(
    model_id="fraud-detector-v1",
    model_type=ModelTypes.BINARY,
    environment=Environments.PRODUCTION,
    dataframe=df_logs,
    prediction_id_column_name="transaction_id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual"
)
📜 License
This project is licensed under the MIT License.
See the LICENSE file for details.

🙌 Contributing
Contributions, bug reports, and improvements are welcome!
Feel free to fork the repo and submit a pull request.



Tools used: AWS S3, MLflow, Arize AI, Python









A

