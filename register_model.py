import mlflow
import mlflow.sklearn
import json
import joblib
import sys
import io

# Fix Windows stdout encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# MLflow tracking server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("CreditCard_Fraud_Detection")

# Load the trained model
model_path = "model.pkl"
model = joblib.load(model_path)

# Load evaluation metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Start MLflow run
with mlflow.start_run(run_name="Model Logging") as run:
    # Log model artifact and register it
    model_uri = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CreditCardFraudModel"  # <- REGISTRATION HAPPENS HERE
    )

    # Log metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    # Log files as artifacts (optional)
    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact("model.pkl")

    print(f"\n✅ Model logged and registered in MLflow as 'CreditCardFraudModel'")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Model URI: {model_uri}")
