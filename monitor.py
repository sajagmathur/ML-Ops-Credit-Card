import os
import pandas as pd
import snowflake.connector
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently import Dataset, DataDefinition
import json
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  
# Load Snowflake credentials from environment variables

import mlflow
import snowflake.connector

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)



# Load config from environment
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')
schema = os.getenv('SNOWFLAKE_SCHEMA')

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Monitoring_Experiments_V1")

def fetch_from_snowflake(query):
    conn = snowflake.connector.connect(
        user=user, password=password,
        account=account, warehouse=warehouse,
        database=database, schema=schema
    )
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()
    return df

def load_champion_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name = 'CreditCardFraudModel'")
    for v in versions:
        if v.current_stage.lower() == "production" and v.tags.get("role") == "champion" and v.tags.get("status") == "production":
            print(f"Loaded model version: {v.version}")
            return mlflow.sklearn.load_model(f"models:/CreditCardFraudModel/Production")
    raise Exception("No Champion model found in Production.")

def calc_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "MatthewsCorrcoef": matthews_corrcoef(y_true, y_pred),
    }

def main():
    model = load_champion_model()

    ref = fetch_from_snowflake("SELECT * FROM CREDITCARD_REFERENCE.PUBLIC.CREDITCARD_REFERENCE")
    cur = fetch_from_snowflake("SELECT * FROM CREDITCARD.PUBLIC.BATCH_PREDICTIONS")
    target = "CLASS"

    ref["prediction"] = model.predict(ref.drop(columns=[target]))
    cur["prediction"] = model.predict(cur.drop(columns=[target]))

    dd = DataDefinition(
        numerical_columns=[col for col in ref.drop(columns=[target]).columns],
        categorical_columns=None,
        target_columns=[target],
        prediction_columns=["prediction"],
        task="classification"
    )

    ds_ref = Dataset.from_pandas(ref, data_definition=dd)
    ds_cur = Dataset.from_pandas(cur, data_definition=dd)

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=ds_ref, current_data=ds_cur)
    report.save_html("evidently_report.html")

    ref_metrics = calc_metrics(ref[target], ref["prediction"])
    cur_metrics = calc_metrics(cur[target], cur["prediction"])

    with open("metrics.json", "w") as f:
        json.dump({"Reference": ref_metrics, "Current": cur_metrics}, f, indent=4)

    degraded = [m for m in ref_metrics if cur_metrics[m] < 0.9 * ref_metrics[m]]
    decision = "YES" if degraded else "NO"
    rationale = f"Degraded metrics: {', '.join(degraded)}" if degraded else "All metrics within threshold."

    pd.DataFrame({
        "Retraining_Decision": [decision],
        "Rationale": [rationale]
    }).to_csv("Retrain.csv", index=False)

    with mlflow.start_run(run_name="Monitoring_Champion") as run:
        mlflow.log_artifact("evidently_report.html")
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("Retrain.csv")
        for k,v in cur_metrics.items():
            mlflow.log_metric(f"Current_{k}", v)
        for k,v in ref_metrics.items():
            mlflow.log_metric(f"Reference_{k}", v)
        mlflow.set_tag("Retrain_Decision", decision)
        mlflow.set_tag("Rationale", rationale)
        mlflow.set_tag("Model_Stage", "Production")
        mlflow.set_tag("Model_Role", "Champion")

    print("Monitoring complete. Report and metrics logged to MLflow.")

if __name__ == "__main__":
    main()
