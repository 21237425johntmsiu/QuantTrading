"""Save StandardScaler parameters from the trained model for use in Rust."""

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from sklearn.preprocessing import StandardScaler

from db_env import db_config

DB_CONFIG = db_config()
BUY_TABLE = "btcusd_17280_BUY720_336_5s"


def fetch_data(query):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    conn.close()
    return df


# Compute periods (same as in the signal script)
def get_periods():
    periods = [12]
    current = 12
    step = 12
    total = int(3600 * 24 / 5) * 1
    target = 12 * 60
    while current < total:
        current += step
        if current <= total:
            periods.append(current)
        if current >= target:
            step = 60
    return periods

def get_columnB():
    periods = [12]
    current = 12
    step = 12
    total = int(3600 * 4 / 5)
    target = 12 * 60
    while current < total:
        current += step
        if current <= total:
            periods.append(current)
        if current >= target:
            step = 60
    return [f'change{period}' for period in periods]


periods = get_periods()
columnB = get_columnB()

# Load training data (720 rows used in pre_train)
query = f"""
SELECT DISTINCT * FROM (
    SELECT DISTINCT * FROM {BUY_TABLE}
    ORDER BY open_timestamp DESC
    LIMIT 720
) AS last_rows
ORDER BY open_timestamp ASC;
"""
df = fetch_data(query)

# Model1 features: the pre-train code inserts 3 columns (change_int, Y_change15, change15_int)
# at positions 4,5,6 before taking columns[14:]. So on the raw table (no inserts),
# features start at column 11 (skipping 11 fixed columns: snapshotTime, timestamp,
# close, volume, change, buy_sell, holding_time, open_timestamp, close_timestamp,
# open_date, close_date)
feature_cols = df.columns[11:].tolist()
# Model2 features: columnB intersection
columnB_cols = columnB

# Fit scalers on the actual data used during training
if not df.empty:
    scaler1 = StandardScaler()
    scaler1.fit(df[feature_cols].values)
    np.save('checkpoints/scaler_A1_mean.npy', scaler1.mean_)
    np.save('checkpoints/scaler_A1_scale.npy', scaler1.scale_)
    np.savetxt('checkpoints/scaler_A1_mean.csv', scaler1.mean_, delimiter=',')
    np.savetxt('checkpoints/scaler_A1_scale.csv', scaler1.scale_, delimiter=',')
    print(f"Model1 scaler: {len(scaler1.mean_)} features, mean={scaler1.mean_[:3]}...")

    scaler2 = StandardScaler()
    available_cols = [c for c in columnB_cols if c in df.columns]
    scaler2.fit(df[available_cols].values)
    np.save('checkpoints/scaler_A2_mean.npy', scaler2.mean_)
    np.save('checkpoints/scaler_A2_scale.npy', scaler2.scale_)
    np.savetxt('checkpoints/scaler_A2_mean.csv', scaler2.mean_, delimiter=',')
    np.savetxt('checkpoints/scaler_A2_scale.csv', scaler2.scale_, delimiter=',')
    print(f"Model2 scaler: {len(scaler2.mean_)} features, mean={scaler2.mean_[:3]}...")

    print("Scaler parameters saved to checkpoints/")
else:
    print("No training data loaded!")
