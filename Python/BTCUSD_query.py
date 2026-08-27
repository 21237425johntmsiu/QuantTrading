import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime
import time

from db_env import db_config

DB_CONFIG = db_config()

def fetch_data_from_database(query, db_config=None, return_type='dataframe'):
    """
    Fetch data from TimescaleDB (PostgreSQL) with error handling.

    Parameters:
    - query: SQL query string to execute
    - db_config: Dictionary containing database connection parameters
    - return_type: 'dataframe' (default) or 'raw'

    Returns:
    - Depending on return_type: DataFrame or raw cursor
    """
    if db_config is None:
        db_config = DB_CONFIG

    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query)

        if return_type == 'raw':
            return cursor

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)

        # safety: catch any unquoted lowercased column aliases
        if 'snapshottime' in df.columns:
            df = df.rename(columns={'snapshottime': 'snapshotTime'})

        # Convert timezone-aware snapshotTime to naive HKT (Asia/Shanghai)
        # so .values[-1] display doesn't silently convert to UTC.
        if 'snapshotTime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['snapshotTime']):
            if df['snapshotTime'].dt.tz is not None:
                df['snapshotTime'] = df['snapshotTime'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

        return df

    except Exception as e:
        print(f"Database error: {e}")
        raise
    finally:
        if connection:
            connection.close()

def create_signal_tables(db_config=None):
    """Create signal tables if they don't exist."""
    if db_config is None:
        db_config = DB_CONFIG
    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        for table_name in ["btcusd_17280_BUY720_336_5s_singal", "btcusd_17280_SELL720_336_5s_singal"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT,
                    check_buy FLOAT8,
                    check_buy2 FLOAT8,
                    open_timestamp BIGINT
                );
            """)
        connection.commit()
    except Exception as e:
        print(f"Error creating signal tables: {e}")
    finally:
        if connection:
            connection.close()

query = """
SELECT "snapshotTime",
       extract(epoch from "snapshotTime")::bigint AS timestamp,
       open, high, low, close, volume
FROM (
    SELECT * FROM BTCUSD_5s
    ORDER BY "snapshotTime" DESC
    LIMIT 200000
) AS sub
ORDER BY "snapshotTime" ASC;
"""
train = fetch_data_from_database(query)

print(train.timestamp.diff(1).value_counts())
print()

table_name = f"BTCUSD_17280_BUY720_336_5s"

query = f"""
SELECT DISTINCT * FROM (
    SELECT DISTINCT * FROM {table_name}
    ORDER BY open_timestamp DESC
    LIMIT 10000
) AS last_rows
ORDER BY open_timestamp ASC;
"""
train = fetch_data_from_database(query)
print(train.open_timestamp.diff(1).value_counts())
print('last_open:',train['open_date'].values[-1])
print('last_close:',train['close_date'].values[-1])
print(pd.to_datetime(train['open_timestamp'].values, unit='s') + pd.Timedelta(hours=8))


