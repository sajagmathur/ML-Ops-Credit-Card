import mlflow
from mlflow.tracking import MlflowClient

# Define the metrics that challenger must beat champion on to become champion
METRICS_TO_COMPARE = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Matthews Corrcoef']

def get_model_versions(client, model_name):
    return client.search_model_versions(f"name='{model_name}'")

def get_model_version_metrics(client, model_name, version):
    # Fetch run metrics for this model version
    mv = client.get_model_version(name=model_name, version=version)
    run_id = mv.run_id
    run = client.get_run(run_id)
    return run.data.metrics

def get_model_version_by_tag(client, model_name, tag_key, tag_value):
    versions = get_model_versions(client, model_name)
    for v in versions:
        tags = v.tags
        if tags.get(tag_key) == tag_value:
            return v
    return None

def better_than(metrics_a, metrics_b):
    """Return True if metrics_a is better than metrics_b on majority of key metrics."""
    better_count = 0
    for metric in METRICS_TO_COMPARE:
        a_val = metrics_a.get(metric)
        b_val = metrics_b.get(metric)
        if a_val is None or b_val is None:
            continue
        if a_val > b_val:
            better_count += 1
    # challenger must be better on > half of the metrics
    return better_count > len(METRICS_TO_COMPARE) / 2

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    model_name = "CreditCardFraudModel"

    challenger_version = get_model_version_by_tag(client, model_name, "role", "challenger")
    if not challenger_version:
        print("❌ No challenger model found.")
        return

    print(f"ℹ️ Challenger model found: Version {challenger_version.version}, Run ID: {challenger_version.run_id}")

    champion_version = get_model_version_by_tag(client, model_name, "status", "production")

    if not champion_version:
        print("⚠️ No champion model found in production. Promoting challenger to production.")
        client.set_model_version_tag(model_name, challenger_version.version, "status", "production")
        client.set_model_version_tag(model_name, challenger_version.version, "role", "champion")
        print(f"🚀 Challenger version {challenger_version.version} promoted to production as champion.")
        return

    print(f"ℹ️ Champion model found: Version {champion_version.version}, Run ID: {champion_version.run_id}")

    challenger_metrics = get_model_version_metrics(client, model_name, challenger_version.version)
    champion_metrics = get_model_version_metrics(client, model_name, champion_version.version)
    
    # Print metrics side-by-side
    print("\n📊 Metrics Comparison:")
    print(f"{'Metric':<20} {'Challenger':<15} {'Champion':<15}")
    print("-" * 50)
    for metric in METRICS_TO_COMPARE:
        challenger_val = challenger_metrics.get(metric, 'N/A')
        champion_val = champion_metrics.get(metric, 'N/A')
        print(f"{metric:<20} {str(challenger_val):<15} {str(champion_val):<15}")
    if better_than(challenger_metrics, champion_metrics):
        print(f"🚀 Challenger version {challenger_version.version} is better than champion version {champion_version.version}. Promoting challenger.")
        # Archive old champion by tag update
        client.set_model_version_tag(model_name, champion_version.version, "status", "archived")
        client.set_model_version_tag(model_name, champion_version.version, "role", "archived")
        # Promote challenger
        client.set_model_version_tag(model_name, challenger_version.version, "status", "production")
        client.set_model_version_tag(model_name, challenger_version.version, "role", "champion")
        print(f"✅ Promotion complete.")
    else:
        print(f"⚠️ Challenger version {challenger_version.version} did NOT beat champion. No changes made.")

if __name__ == "__main__":
    main()
