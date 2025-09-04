import os
import pandas as pd
import snowflake.connector
from evidently import ColumnMapping
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from sklearn.metrics import f1_score
import json

# Load Snowflake credentials from environment variables
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')  # 'CREDITCARD'
schema = os.getenv('SNOWFLAKE_SCHEMA')      # 'PUBLIC'

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
    # Step 1: Load reference and current batch predictions data
    print("📥 Fetching reference data...")
    reference_data = fetch_data_from_snowflake(REFERENCE_TABLE)
    print(f"Reference data shape: {reference_data.shape}")

    print("📥 Fetching current batch predictions data...")
    current_data = fetch_data_from_snowflake(CURRENT_DATA_TABLE)
    print(f"Current data shape: {current_data.shape}")

    # Step 2: Setup ColumnMapping for evidently
    # Adjust column names if necessary; assumes:
    # reference_data: original data with 'CLASS' column as target
    # current_data: predictions data including 'CLASS' (if available), 'PREDICTION', 'PREDICTION_PROB'
    column_mapping = ColumnMapping()
    column_mapping.target = "CLASS"
    column_mapping.prediction = "PREDICTION"
    # Assuming features are all except these columns
    feature_columns = [col for col in reference_data.columns if col not in ['ID', 'CLASS']]
    column_mapping.numerical_features = feature_columns

    # Step 3: Generate data drift report
    print("📊 Generating data drift report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    data_drift_metrics = data_drift_report.as_dict()

    # Extract overall drift score for dataset
    data_drift_score = data_drift_metrics['metrics'][0]['result']['dataset_drift']

    print(f"Data Drift Score: {data_drift_score}")

    # Step 4: Generate classification performance report
    print("📊 Generating classification performance report...")
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    classification_metrics = classification_report.as_dict()

    # Extract F1 scores
    def get_f1_score(metrics_dict, dataset_label):
        for metric in metrics_dict['metrics']:
            if metric['metric_name'] == 'ClassificationPreset':
                f1 = metric['result']['metrics']['F1']['value']
                return f1
        return None

    f1_reference = get_f1_score(classification_metrics, 'reference')
    f1_current = get_f1_score(classification_metrics, 'current')

    print(f"Reference F1 Score: {f1_reference}")
    print(f"Current F1 Score: {f1_current}")

    # Step 5: Decide retraining
    # Simple heuristic:
    # Retrain if data drift > 0.3 or F1 drop > 0.1
    retrain = 'N'
    rationale = "Data within acceptable range; no significant drift or performance degradation."

    if data_drift_score is not None and data_drift_score > 0.3:
        retrain = 'Y'
        rationale = f"High data drift detected (drift score={data_drift_score:.2f}), retraining recommended."
    elif f1_reference is not None and f1_current is not None and (f1_reference - f1_current) > 0.1:
        retrain = 'Y'
        rationale = f"F1 score dropped significantly from {f1_reference:.3f} to {f1_current:.3f}, retraining recommended."

    print(f"🔔 Retraining Decision: {retrain}")
    print(f"📝 Rationale: {rationale}")

    # Step 6: Save decision and rationale to CSV
    decision_df = pd.DataFrame({'Decision': [retrain], 'Rationale': [rationale]})
    decision_df.to_csv(RETRAINING_DECISION_PATH, index=False)
    print(f"✅ Retraining decision and rationale saved to {RETRAINING_DECISION_PATH}")

if __name__ == "__main__":
    main()
