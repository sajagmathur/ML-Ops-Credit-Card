import os
import pandas as pd
import snowflake.connector
import mlflow
from mlflow.tracking import MlflowClient
import sys
import io

# Fix Windows stdout encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Lo#ad Snowflake credentials from environment variables
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# MLflow tracking URI and model name
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "CreditCardFraudModel"

# Snowflake tables
BATCH_INPUT_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.CREDITCARD_BATCH_INPUTS"
BATCH_PREDICTIONS_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.BATCH_PREDICTIONS"


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
    print(f"📥 Fetching batch data from Snowflake table: {BATCH_INPUT_TABLE}")
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    try:
        query = f"SELECT * FROM {BATCH_INPUT_TABLE}"
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        print(f"✅ Fetched {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    finally:
        cursor.close()
        conn.close()


def get_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.tags.get("role") == "champion" and v.tags.get("status") == "production":
            print(f"🎯 Loading champion model version: {v.version}")
            model_uri = f"models:/{MODEL_NAME}/{v.version}"
            model = mlflow.sklearn.load_model(model_uri)
            return model

    raise Exception("❌ No champion model found in production.")


def generate_predictions(df, model):
    # Dynamically add ID if missing
    if 'ID' not in df.columns:
        df.insert(0, 'ID', range(1, len(df) + 1))

    ids = df['ID']
    has_class = 'CLASS' in df.columns
    features = df.drop(columns=['ID'] + (['CLASS'] if has_class else []))

    print(f"🔍 Generating predictions for {features.shape[0]} records...")

    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = [None] * len(preds)

    # Construct full output
    result_df = df.copy()
    result_df['PREDICTION'] = preds
    result_df['PREDICTION_PROB'] = probs

    return result_df


def save_predictions_to_snowflake(df):
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    try:
        print(f"🧹 Truncating table {BATCH_PREDICTIONS_TABLE} before inserting new predictions.")
        cursor.execute(f"TRUNCATE TABLE {BATCH_PREDICTIONS_TABLE}")
        conn.commit()

        print(f"⬆️ Inserting {len(df)} prediction rows into {BATCH_PREDICTIONS_TABLE}...")

        # Prepare insert statement
        cols = list(df.columns)
        col_str = ', '.join(cols)
        val_placeholders = ', '.join(['%s'] * len(cols))
        insert_query = f"INSERT INTO {BATCH_PREDICTIONS_TABLE} ({col_str}) VALUES ({val_placeholders})"

        # Convert dataframe to list of tuples
        data_tuples = [tuple(x) for x in df.to_numpy()]

        # Execute batch insert
        cursor.executemany(insert_query, data_tuples)
        conn.commit()
        print("✅ Predictions saved to Snowflake table successfully.")
    finally:
        cursor.close()
        conn.close()


def main():
    batch_df = fetch_batch_data()
    model = get_champion_model()
    predictions_df = generate_predictions(batch_df, model)
    save_predictions_to_snowflake(predictions_df)
    print("🏁 Batch inference pipeline completed successfully.")


if __name__ == "__main__":
    main()
