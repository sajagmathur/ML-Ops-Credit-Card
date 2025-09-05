import os
import snowflake.connector
import pandas as pd
from datetime import datetime,timezone
import sys
import io

# Fix stdout encoding for Windows runners, ignore if not needed
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Connect to Snowflake using environment variables
conn = snowflake.connector.connect(
    user=os.environ['SNOWFLAKE_USER'],
    password=os.environ['SNOWFLAKE_PASSWORD'],
    account=os.environ['SNOWFLAKE_ACCOUNT'],
    warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
    database=os.environ['SNOWFLAKE_DATABASE'],
    schema=os.environ['SNOWFLAKE_SCHEMA']
)

cursor = conn.cursor()

# Select the latest retraining decision ordered by UPDATED_AT (timestamp column)
query = """
SELECT * FROM CREDITCARD.PUBLIC.RETRAIN
ORDER BY UPDATED_AT DESC
LIMIT 1
"""

df = cursor.execute(query).fetch_pandas_all()

if df.empty:
    print("ℹ️ No records found in the retrain table.")
else:
    decision = df.iloc[0]['RETRAINING_DECISION'].strip().upper()
    rationale = df.iloc[0]['RATIONALE']

    if decision == "YES":
        print("🔁 Retraining triggered based on decision YES.")

        # 1. Write retrain.txt
        with open("retrain.txt", "w") as f:
            f.write("Retraining")

        # 2. Update the Snowflake table to mark retraining done
        update_query = f"""
        UPDATE CREDITCARD.PUBLIC.RETRAIN
        SET RETRAINING_DECISION = 'NO',
            RATIONALE = 'Retrained at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC',
            UPDATED_AT = CURRENT_TIMESTAMP()
        WHERE RETRAINING_DECISION = 'YES'
        """
        cursor.execute(update_query)
        conn.commit()

        print("✅ Retraining flag updated in Snowflake.")
    else:
        print("⏭️ No retraining required.")

cursor.close()
conn.close()
