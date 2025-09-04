import mlflow
import mlflow.sklearn
import json
import joblib

# MLflow tracking server URI
mlflow.set_tracking_uri("http://192.168.29.15:5000/")
mlflow.set_experiment("CreditCard_Fraud_Detection")

# Load the trained model
model_path = "model.pkl"
model = joblib.load(model_path)

# Load evaluation metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Start MLflow run
with mlflow.start_run(run_name="Logged_Model_From_File") as run:
    # Log model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    # Log files as artifacts (optional: helpful for reproducibility)
    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact("model.pkl")

    print(f"\n✅ Model and metrics successfully logged to MLflow run: {run.info.run_id}")
