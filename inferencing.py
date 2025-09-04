import os
import pandas as pd
import snowflake.connector
import mlflow
from mlflow.tracking import MlflowClient

# Load Snowflake credentials from environment variables
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')  # e.g., 'CREDITCARD'
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')      # e.g., 'PUBLIC'

# MLflow tracking URI and model name
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # Adjust if needed
MODEL_NAME = "CreditCardFraudModel"

# Snowflake batch input table
BATCH_INPUT_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.CREDITCARD_BATCH_INPUTS"

# Local save path for CSV predictions
LOCAL_SAVE_PATH = r"C:\Users\sajag\Desktop\Data_Split\batch_predictions.csv"


def get_snowflake_connection():
    return snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )


def fetch_batch_data():
    print(f"Fetching batch data from Snowflake table: {BATCH_INPUT_TABLE}")
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    try:
        query = f"SELECT * FROM {BATCH_INPUT_TABLE}"
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        print(f"Fetched {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    finally:
        cursor.close()
        conn.close()


def get_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Find model with tag role='champion' and status='production'
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.tags.get("role") == "champion" and v.tags.get("status") == "production":
            print(f"Loading champion model version: {v.version}")
            model_uri = f"models:/{MODEL_NAME}/{v.version}"
            model = mlflow.sklearn.load_model(model_uri)
            return model

    raise Exception("No champion model found in production.")


def generate_predictions(df, model):
    if 'ID' not in df.columns:
        raise Exception("Input batch data must contain an 'ID' column.")

    ids = df['ID']
    X = df.drop(columns=['ID'])

    print(f"Generating predictions for {X.shape[0]} records...")

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]  # Prob for positive class
    else:
        probs = [None] * len(preds)

    pred_df = X.copy()
    pred_df['ID'] = ids
    pred_df['PREDICTION'] = preds
    pred_df['PREDICTION_PROB'] = probs

    # Reorder to have ID first
    cols = ['ID'] + [col for col in pred_df.columns if col != 'ID']
    pred_df = pred_df[cols]

    return pred_df


def save_predictions_to_csv(df, filename=LOCAL_SAVE_PATH):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Predictions saved locally to {filename}")


def main():
    batch_df = fetch_batch_data()
    model = get_champion_model()
    predictions_df = generate_predictions(batch_df, model)
    save_predictions_to_csv(predictions_df)
    print("🏁 Batch inference pipeline completed successfully.")


if __name__ == "__main__":
    main()
