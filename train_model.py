import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
import joblib

# Snowflake credentials
account = 'onmhvte-rm57820'
user = 'SAJAGMATHUR'
password = 'Thati10pur@719'
warehouse = 'COMPUTE_WH'
database = 'CREDITCARD'
schema = 'PUBLIC'

def fetch_data_from_snowflake():
    # Establish connection to Snowflake
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    cur = conn.cursor()
    # Replace 'your_table' with your actual table name
    cur.execute("SELECT * FROM CREDITCARD.PUBLIC.CREDITCARD")
    df = cur.fetch_pandas_all()
    conn.close()
    return df

def main():
    # Fetch data from Snowflake
    data = fetch_data_from_snowflake()
    print("Data loaded from Snowflake. Shape:", data.shape)

    # Prepare features and target
    X = data.drop(['CLASS'], axis=1)
    y = data['CLASS']
    print("\nFeature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    # Train-test split
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Split data into train and test sets:")
    print("xTrain:", xTrain.shape, "xTest:", xTest.shape)

    # Train Random Forest Model
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    print("\nRandom Forest model trained.")

    # Predictions & evaluation
    yPred = rfc.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    precision = precision_score(yTest, yPred)
    recall = recall_score(yTest, yPred)
    f1 = f1_score(yTest, yPred)
    mcc = matthews_corrcoef(yTest, yPred)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(yTest, yPred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Save the trained model
    model_filename = "model.pkl"
    joblib.dump(rfc, model_filename)
    print(f"\nModel saved to {model_filename}")


    print("Model Training Complete, Saving Dataset as Reference to Snowflake")
    

if __name__ == "__main__":
    main()