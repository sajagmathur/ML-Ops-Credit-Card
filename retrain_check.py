import os
import snowflake.connector
import pandas as pd
from datetime import datetime, timezone
import sys
import io
import numpy as np
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
"""

df = cursor.execute(query).fetch_pandas_all()

if df.empty:
    print("ℹ️ No records found in the retrain table.")
    # Always set output for GitHub Actions
    with open(os.environ.get('GITHUB_OUTPUT', 'github_output.txt'), 'a') as f:
        f.write("retrain=false\n")
else:
    decision_value = df.iloc[0]['RETRAINING_DECISION']
    rationale = df.iloc[0]['RATIONALE']

    # Normalize decision_value to "YES" or "NO"
    print(f"Raw decision_value from Snowflake: {decision_value} (type: {type(decision_value)})")
    if isinstance(decision_value, (bool, np.bool_)):
        decision = "YES" if decision_value else "NO"
    elif isinstance(decision_value, (int, float)):
        decision = "YES" if decision_value == 1 else "NO"
    elif isinstance(decision_value, str):
        decision = decision_value.strip().upper()
        if decision in ["TRUE", "T", "1"]:
            decision = "YES"
        elif decision in ["FALSE", "F", "0"]:
            decision = "NO"
    else:
        decision = str(decision_value).strip().upper()

    if decision == "YES":
        print("🔁 Retraining triggered based on decision YES.")
        with open(os.environ.get('GITHUB_OUTPUT', 'github_output.txt'), 'a') as f:
            f.write("retrain=true\n")

        # 2. Update the Snowflake table to mark retraining done
        update_query = f"""
        UPDATE CREDITCARD.PUBLIC.RETRAIN
        SET RETRAINING_DECISION = 'NO',
            RATIONALE = 'Retrained at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC'
        WHERE RETRAINING_DECISION IN ('YES', 'TRUE', 'T', '1')
        """
        cursor.execute(update_query)
        conn.commit()

        print("✅ Retraining flag updated in Snowflake.")
    else:
        print("⏭️ No retraining required.")
        with open(os.environ.get('GITHUB_OUTPUT', 'github_output.txt'), 'a') as f:
            f.write("retrain=false\n")

cursor.close()
conn.close()
