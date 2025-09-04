import mlflow
import mlflow.sklearn
import json
import joblib
import sys
import io
from mlflow.tracking import MlflowClient  # <-- Needed for transition

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
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CreditCardFraudModel"
    )

    # Log metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    # Log files as artifacts (optional)
    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact("model.pkl")

    print(f"\n✅ Model logged and registered in MLflow as 'CreditCardFraudModel'")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Model URI: {model_info.model_uri}")

    # --- Transition to STAGING (mark as CHALLENGER) ---
    client = MlflowClient()
    model_name = model_info.name
    model_version = model_info.version

    # Transition to STAGING stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging"
    )

    # Optional: Add a tag to mark it as "challenger"
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="role",
        value="challenger"
    )

    print(f"🚀 Model version {model_version} transitioned to STAGING (as Challenger)")
