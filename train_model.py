import os
import json
import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
import mlflow
import mlflow.sklearn  # needed for logging sklearn models

# Load credentials from environment variables
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')
schema = os.getenv('SNOWFLAKE_SCHEMA')

# MLflow settings
mlflow.set_tracking_uri("http://192.168.29.15:5000")  # Adjust if needed
mlflow.set_experiment("CreditCard Fraud Detection")

def fetch_data_from_snowflake():
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM CREDITCARD.PUBLIC.CREDITCARD")
    df = cur.fetch_pandas_all()
    conn.close()
    return df

def copy_reference_table():
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database='CREDITCARD_REFERENCE',
        schema='PUBLIC'
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE OR REPLACE TABLE CREDITCARD_REFERENCE.PUBLIC.CREDITCARD_REFERENCE AS
        SELECT * FROM CREDITCARD.PUBLIC.CREDITCARD
    """)
    print("✅ Reference table copied to CREDITCARD_REFERENCE.PUBLIC.CREDITCARD_REFERENCE.")
    conn.close()

def main():
    data = fetch_data_from_snowflake()
    print("✅ Data loaded from Snowflake. Shape:", data.shape)

    X = data.drop(['CLASS'], axis=1)
    y = data['CLASS']
    print(f"🎯 Features shape: {X.shape}, Target shape: {y.shape}")

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Data split into train/test sets.")

    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier()
        model.fit(xTrain, yTrain)
        print("✅ Model trained.")

        # Evaluate
        yPred = model.predict(xTest)
        metrics = {
            'accuracy': accuracy_score(yTest, yPred),
            'precision': precision_score(yTest, yPred),
            'recall': recall_score(yTest, yPred),
            'f1_score': f1_score(yTest, yPred),
            'matthews_corrcoef': matthews_corrcoef(yTest, yPred)
        }

        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            print(f"{key}: {value:.4f}")

        print("\n📉 Confusion Matrix:")
        print(confusion_matrix(yTest, yPred))

        # Save metrics as local file too
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics.json")

        # Log model to MLflow and register it
        mlflow.sklearn.log_model(model, "model", registered_model_name="creditcard-rf-model")
        print("✅ Model logged and registered to MLflow.")

    # Copy reference data
    print("📤 Copying reference data to Snowflake...")
    copy_reference_table()

    print("\n🏁 All steps completed successfully.")

if __name__ == "__main__":
    main()
