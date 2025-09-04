import mlflow
from mlflow.tracking import MlflowClient
import sys
import os
import io

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# MLflow tracking server URI (adjust if needed)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

model_name = "CreditCardFraudModel"
client = MlflowClient()

def get_production_model_version(model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    production_versions = [v for v in versions if v.current_stage == "Production"]
    if not production_versions:
        return None
    # Return latest production version
    return max(production_versions, key=lambda v: int(v.version))

def get_challenger_model_version(model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        model_version = client.get_model_version(name=model_name, version=v.version)
        tags = model_version.tags  # Correct way to get tags
        if tags.get("role") == "challenger" and tags.get("status") == "staging":
            return v
    return None

def fetch_metrics_from_run(run_id):
    try:
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        print(f"❌ Error fetching run metrics for run_id={run_id}: {e}")
        return {}

def is_challenger_better(challenger_metrics, champion_metrics):
    # Challenger must beat champion on Recall and F1 Score strictly
    if challenger_metrics.get('Recall', 0) <= champion_metrics.get('Recall', 0):
        print("❌ Challenger Recall not better")
        return False
    if challenger_metrics.get('F1 Score', 0) <= champion_metrics.get('F1 Score', 0):
        print("❌ Challenger F1 Score not better")
        return False

    # Challenger Precision and Matthews Corrcoef must be >= champion's
    for metric in ['Precision', 'Matthews Corrcoef']:
        challenger_val = challenger_metrics.get(metric, 0)
        champion_val = champion_metrics.get(metric, 0)
        if challenger_val < champion_val:
            print(f"❌ Challenger {metric} worse ({challenger_val:.4f} < {champion_val:.4f})")
            return False

    # Accuracy is not strictly checked here

    print("✅ Challenger is better based on defined criteria")
    return True

def main():
    champion_version = get_production_model_version(model_name)
    challenger_version = get_challenger_model_version(model_name)

    if not challenger_version:
        print("⚠️ No challenger model found with role='challenger' and status='staging'. Exiting.")
        return

    print(f"ℹ️ Challenger model found: Version {challenger_version.version}, Run ID: {challenger_version.run_id}")

    challenger_metrics = fetch_metrics_from_run(challenger_version.run_id)

    if champion_version is None:
        print("⚠️ No champion model found in Production. Promoting challenger to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version.version,
            stage="Production",
            archive_existing_versions=False
        )
        client.set_model_version_tag(model_name, challenger_version.version, "role", "champion")
        client.set_model_version_tag(model_name, challenger_version.version, "status", "production")
        print(f"🎉 Challenger version {challenger_version.version} promoted to Production!")
        return

    print(f"ℹ️ Champion model found: Version {champion_version.version}, Run ID: {champion_version.run_id}")

    champion_metrics = fetch_metrics_from_run(champion_version.run_id)

    if is_challenger_better(challenger_metrics, champion_metrics):
        print(f"🎉 Challenger model version {challenger_version.version} is better than champion version {champion_version.version}")

        # Archive champion
        client.transition_model_version_stage(
            name=model_name,
            version=champion_version.version,
            stage="Archived"
        )
        print(f"📦 Archived champion version {champion_version.version}")

        # Promote challenger
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version.version,
            stage="Production",
            archive_existing_versions=False
        )
        print(f"🚀 Promoted challenger version {challenger_version.version} to Production")

        # Update tags for challenger to champion
        client.set_model_version_tag(model_name, challenger_version.version, "role", "champion")
        client.set_model_version_tag(model_name, challenger_version.version, "status", "production")

    else:
        print(f"⚠️ Challenger model version {challenger_version.version} is NOT better than champion version {champion_version.version}")
        print("⏳ Keeping current champion and challenger remains staging.")

if __name__ == "__main__":
    main()
