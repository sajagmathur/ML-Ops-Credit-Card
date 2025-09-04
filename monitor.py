import os
import pandas as pd
import snowflake.connector
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently import Dataset, DataDefinition
import json

# Load Snowflake credentials from environment variables
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')  # e.g. 'CREDITCARD'
schema = os.getenv('SNOWFLAKE_SCHEMA')      # e.g. 'PUBLIC'

# Snowflake tables
REFERENCE_TABLE = 'CREDITCARD_REFERENCE.PUBLIC.CREDITCARD_REFERENCE'
CURRENT_DATA_TABLE = 'CREDITCARD.PUBLIC.BATCH_PREDICTIONS'

# Path to save retraining decision CSV
RETRAINING_DECISION_PATH = r"C:\Users\sajag\Desktop\Data_Split\retraining_decision.csv"

def fetch_data_from_snowflake(table_name):
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    cursor = conn.cursor()
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    return df

def main():
    print("📥 Fetching reference data...")
    reference_data = fetch_data_from_snowflake(REFERENCE_TABLE)
    print(f"Reference data shape: {reference_data.shape}")

    print("📥 Fetching current batch predictions data...")
    current_data = fetch_data_from_snowflake(CURRENT_DATA_TABLE)
    print(f"Current data shape: {current_data.shape}")

    # Define your columns explicitly — adjust to your data's columns and types
    target_column = "CLASS"
    prediction_column = "PREDICTION"
    # For features, assume all except ID, target, prediction columns
    feature_columns = [col for col in reference_data.columns if col not in ['ID', target_column, prediction_column]]

    # Define numerical features - here assuming all feature_columns are numerical
    # Adjust if you have categorical features
    data_definition = DataDefinition(
        numerical_columns=feature_columns,
        classification=[ClassificationPreset(target=target_column, prediction=prediction_column)]
    )

    # Create Evidently Dataset objects
    ref_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)
    cur_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)

    print("📊 Generating data drift report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=ref_dataset, current_data=cur_dataset)
    data_drift_metrics = data_drift_report.as_dict()

    data_drift_score = data_drift_metrics['metrics'][0]['result']['dataset_drift']
    print(f"Data Drift Score: {data_drift_score}")

    print("📊 Generating classification performance report...")
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(reference_data=ref_dataset, current_data=cur_dataset)
    classification_metrics = classification_report.as_dict()

    # Extract F1 scores from classification report
    def get_f1_score(metrics_dict):
        for metric in metrics_dict['metrics']:
            if metric['metric_name'] == 'ClassificationPreset':
                return metric['result']['metrics']['F1']['value']
        return None

    f1_reference = get_f1_score(classification_metrics.get('reference', classification_metrics))
    f1_current = get_f1_score(classification_metrics.get('current', classification_metrics))

    print(f"Reference F1 Score: {f1_reference}")
    print(f"Current F1 Score: {f1_current}")

    # Decide retraining based on drift and F1 drop
    retrain = 'N'
    rationale = "Data within acceptable range; no significant drift or performance degradation."

    if data_drift_score is not None and data_drift_score > 0.3:
        retrain = 'Y'
        rationale = f"High data drift detected (score={data_drift_score:.2f}), retraining recommended."
    elif f1_reference is not None and f1_current is not None and (f1_reference - f1_current) > 0.1:
        retrain = 'Y'
        rationale = f"F1 score dropped significantly from {f1_reference:.3f} to {f1_current:.3f}, retraining recommended."

    print(f"🔔 Retraining Decision: {retrain}")
    print(f"📝 Rationale: {rationale}")

    decision_df = pd.DataFrame({'Decision': [retrain], 'Rationale': [rationale]})
    decision_df.to_csv(RETRAINING_DECISION_PATH, index=False)
    print(f"✅ Retraining decision and rationale saved to {RETRAINING_DECISION_PATH}")

if __name__ == "__main__":
    main()
