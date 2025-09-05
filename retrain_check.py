import os
import snowflake.connector
import pandas as pd
from datetime import datetime
import sys
import io
# Fix Windows stdout encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Fetch Snowflake credentials from environment
conn = snowflake.connector.connect(
    user=os.environ['SNOWFLAKE_USER'],
    password=os.environ['SNOWFLAKE_PASSWORD'],
    account=os.environ['SNOWFLAKE_ACCOUNT'],
    warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
    database=os.environ['SNOWFLAKE_DATABASE'],
    schema=os.environ['SNOWFLAKE_SCHEMA']
)

cursor = conn.cursor()

# Fetch the latest retraining decision
query = "SELECT * FROM CREDITCARD.PUBLIC.RETRAIN ORDER BY ROWID DESC LIMIT 1"
df = cursor.execute(query).fetch_pandas_all()

if df.empty:
    print("ℹ️ No records found in the retrain table.")
else:
    decision = df.iloc[0]['RETRAINING_DECISION'].strip().upper()
    rationale = df.iloc[0]['RATIONALE']

    if decision == "YES":
        print("🔁 Retraining triggered based on decision YES.")

        # 1. Write the retrain.txt file
        with open("retrain.txt", "w") as f:
            f.write("Retraining")

        # 2. Update the Snowflake table
        update_query = f"""
        UPDATE CREDITCARD.PUBLIC.RETRAIN
        SET RETRAINING_DECISION = 'NO',
            RATIONALE = 'Retrained at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC'
        WHERE RETRAINING_DECISION = 'YES'
        """
        cursor.execute(update_query)
        conn.commit()

        print("✅ Retraining flag updated in Snowflake.")
    else:
        print("⏭️ No retraining required.")
