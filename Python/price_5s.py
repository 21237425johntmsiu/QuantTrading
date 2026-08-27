import asyncio
import aiohttp
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pymysql
from pymysql.cursors import DictCursor
from db_env import mysql_config
from IPython.display import clear_output
import threading
from termcolor import colored
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import sys
import traceback

def create_table_if_not_exists(df, table_name, db_config):
    """Create table if it doesn't exist, with a unique index on timestamp/from_timestamp."""
    connection = None
    try:
        connection = get_db_connection(db_config, autocommit=True)
        cursor = connection.cursor()
        
        # Check if table exists
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        if cursor.fetchone():
            print(f"   Table '{table_name}' already exists")
            cursor.close()
            connection.close()
            return True
        
        print(f"   Creating table '{table_name}'...")
        # Generate CREATE TABLE statement
        column_defs = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype:
                col_type = 'BIGINT' if df[col].max() > 2147483647 else 'INT'
            elif 'float' in dtype:
                col_type = 'DOUBLE'
            elif 'datetime' in dtype:
                col_type = 'DATETIME'
            else:
                max_len = df[col].astype(str).str.len().max()
                col_type = 'VARCHAR(255)' if max_len <= 255 else 'TEXT'
            column_defs.append(f"`{col}` {col_type}")
        
        create_sql = f"CREATE TABLE `{table_name}` (\n    " + ",\n    ".join(column_defs) + "\n)"
        cursor.execute(create_sql)
        
        # Add unique index on the timestamp column
        if 'from_timestamp' in df.columns:
            cursor.execute(f"CREATE UNIQUE INDEX idx_from_timestamp ON `{table_name}` (from_timestamp)")
            print(f"   Created unique index on from_timestamp")
        elif 'timestamp' in df.columns:
            cursor.execute(f"CREATE UNIQUE INDEX idx_timestamp ON `{table_name}` (timestamp)")
            print(f"   Created unique index on timestamp")
        
        cursor.close()
        connection.close()
        print(f"   Table '{table_name}' created successfully")
        return True
    except Exception as e:
        print(f"   Table creation error: {e}")
        if connection:
            connection.close()
        return False


def get_db_connection(db_config, autocommit=False):
    """Create a database connection, separating connection params from execution params."""
    # Copy config to avoid modifying original
    conn_config = db_config.copy()
    
    # Remove cursorclass if present (it's not a connection parameter)
    cursor_class = conn_config.pop('cursorclass', None)
    
    # Remove autocommit if present (we'll set it explicitly)
    conn_config.pop('autocommit', None)
    
    # Create connection
    connection = pymysql.connect(**conn_config, autocommit=autocommit)
    
    # Set cursorclass if specified
    if cursor_class:
        connection.cursorclass = cursor_class
    
    return connection


def append_to_db(df, table_name, db_config, batch_size=1000):
    """Append rows to MySQL table, skipping duplicates via INSERT IGNORE."""
    connection = None
    try:
        print(f"\n   Preparing to insert {len(df)} rows into '{table_name}'...")
        
        # Ensure table exists
        if not create_table_if_not_exists(df, table_name, db_config):
            return False
        
        # Get connection with autocommit=False for batch operations
        connection = get_db_connection(db_config, autocommit=False)
        cursor = connection.cursor()
        
        # Prepare INSERT IGNORE statement
        columns = [f"`{col}`" for col in df.columns]
        placeholders = ['%s'] * len(columns)
        insert_sql = f"INSERT IGNORE INTO `{table_name}` ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        # Convert DataFrame rows to tuples
        data = []
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                elif isinstance(val, pd.Timestamp):
                    row_data.append(val.to_pydatetime())
                else:
                    row_data.append(val)
            data.append(tuple(row_data))
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            cursor.executemany(insert_sql, batch)
            connection.commit()
            inserted_in_batch = cursor.rowcount
            total_inserted += inserted_in_batch
            print(f"   Batch {i//batch_size + 1}: {len(batch)} rows processed, {inserted_in_batch} inserted")
        
        cursor.close()
        connection.close()
        print(f"   ✅ Successfully inserted {total_inserted} new rows into '{table_name}'")
        return True
    except Exception as e:
        print(f"   ❌ Append error: {e}")
        if connection:
            connection.rollback()
            connection.close()
        return False

        
# -------------------- Synchronous helpers (pandas, DB) --------------------
def df_reshape(length, df1):
    """Resample 1‑minute data to N minutes."""
    df = df1.copy()
    if 'timestamp' not in df.columns and 'snapshotTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['snapshotTime']).astype('int64') // 10**9
    else:
        df['timestamp'] = df['timestamp'].astype('int64')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df_resampled = df.resample(f'{length}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'datetime': 'snapshotTime'}, inplace=True)
    df_resampled['from_timestamp'] = df_resampled['snapshotTime'].apply(lambda x: int(x.timestamp()))
    df_resampled['to_timestamp'] = df_resampled['from_timestamp'] + (length * 60)
    df_resampled['snapshotTime'] = df_resampled['snapshotTime'] + pd.Timedelta(hours=8)
    df_resampled.rename(columns={'from_timestamp': 'timestamp'}, inplace=True)
    return df_resampled[["snapshotTime", "timestamp", "open", "close", "high", "low", "volume"]]

def df_reshape_1s(length, df1):
    """Resample 1‑second data to N seconds (used for 1‑minute aggregation)."""
    df = df1.copy()
    if 'timestamp' not in df.columns and 'snapshotTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['snapshotTime']).astype('int64') // 10**9
    else:
        df['timestamp'] = df['timestamp'].astype('int64')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df_resampled = df.resample(f'{length}s').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'datetime': 'snapshotTime'}, inplace=True)
    df_resampled['from_timestamp'] = df_resampled['snapshotTime'].apply(lambda x: int(x.timestamp()))
    df_resampled['to_timestamp'] = df_resampled['from_timestamp'] + length
    df_resampled['snapshotTime'] = df_resampled['snapshotTime'] + pd.Timedelta(hours=8)
    df_resampled.rename(columns={'from_timestamp': 'timestamp'}, inplace=True)
    return df_resampled[["snapshotTime", "timestamp", "open", "close", "high", "low", "volume"]]

def setup_database_and_table(df, database_name, table_name):
    """Insert/update data in MySQL (synchronous, to be run in thread)."""
    db_config = mysql_config(cursorclass=DictCursor)
    db_config.pop("database", None)  # DB is selected via USE below
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} "
                           f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {database_name}")
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                snapshotTime DATETIME NOT NULL,
                timestamp BIGINT NOT NULL,
                open FLOAT NOT NULL,
                high FLOAT NOT NULL,
                low FLOAT NOT NULL,
                close FLOAT NOT NULL,
                volume INT,
                PRIMARY KEY (snapshotTime),
                INDEX (timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_table_query)

            connection.begin()
            columns = 'snapshotTime, timestamp, open, high, low, close, volume'
            placeholders = ', '.join(['%s'] * 7)
            insert_sql = f"""
            INSERT INTO {table_name} ({columns}) 
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE 
                open=VALUES(open),
                high=VALUES(high),
                low=VALUES(low),
                close=VALUES(close),
                volume=VALUES(volume)
            """
            data_to_insert = [tuple(row) for row in df.values]
            cursor.executemany(insert_sql, data_to_insert)
            connection.commit()
            print(f"Inserted/updated {cursor.rowcount} rows in {table_name}")
    except Exception as e:
        connection.rollback()
        print(f"DB error: {e}")
        raise
    finally:
        connection.close()

def fetch_data_from_database(query, db_config=None, return_type='dataframe'):
    """
    Enhanced function to fetch data from MySQL database with better error handling

    Parameters:
    - query: SQL query string to execute
    - db_config: Dictionary containing database connection parameters
    - return_type: 'dataframe' (default), 'dict', or 'raw' for cursor object

    Returns:
    - Depending on return_type: DataFrame, list of dicts, or raw cursor
    """
    if db_config is None:
        db_config = mysql_config(cursorclass=DictCursor, connect_timeout=10)

    connection = None
    try:
        # Establish connection with timeout
        connection = pymysql.connect(**db_config)

        with connection.cursor() as cursor:
            # Execute query with timeout
            cursor.execute(query)

            if return_type == 'raw':
                return cursor  # Return cursor object for streaming

            results = cursor.fetchall()

            return pd.DataFrame(results)

    except pymysql.MySQLError as e:
        print(f"MySQL Error {e.args[0]}: {e.args[1]}")
        raise
    except Exception as e:
        print(f"General error: {str(e)}")
        raise
    finally:
        if connection:
            connection.close()

def dataframe_to_mysql_direct(df, table_name, if_exists='append'):
    """
    Direct PyMySQL approach - most reliable
    """
    # Database configuration
    db_config = mysql_config(cursorclass=pymysql.cursors.DictCursor)

    connection = None
    cursor = None

    try:
        print(f"📊 Preparing to save {len(df)} rows with {len(df.columns)} columns...")

        # Connect to MySQL
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # Check if table exists
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        table_exists = cursor.fetchone() is not None

        # Handle table existence
        if table_exists:
            if if_exists == 'fail':
                raise Exception(f"Table '{table_name}' already exists")
            elif if_exists == 'replace':
                cursor.execute(f"DROP TABLE {table_name}")
                table_exists = False
                print(f"🗑️  Table '{table_name}' dropped")

        # Create table if needed
        if not table_exists:
            create_table_sql = f"CREATE TABLE {table_name} ("

            # Add columns based on DataFrame
            for col in df.columns:
                dtype = str(df[col].dtype)

                # Map pandas dtype to MySQL dtype
                if 'int' in dtype:
                    mysql_type = "BIGINT"
                elif 'float' in dtype or 'double' in dtype:
                    mysql_type = "DOUBLE"
                elif 'datetime' in dtype:
                    mysql_type = "DATETIME"
                elif 'bool' in dtype:
                    mysql_type = "BOOLEAN"
                else:
                    # For strings and others
                    max_len = df[col].astype(str).str.len().max()
                    if pd.isna(max_len):
                        max_len = 255
                    mysql_type = f"VARCHAR({int(max_len * 1.5)})"

                create_table_sql += f"`{col}` {mysql_type}, "

            create_table_sql = create_table_sql.rstrip(', ') + ")"
            cursor.execute(create_table_sql)
            print(f"📋 Table '{table_name}' created")

        # Prepare insert statement
        columns = [f"`{col}`" for col in df.columns]
        placeholders = ['%s'] * len(columns)

        insert_sql = f"""
            INSERT INTO {table_name}
            ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        # Insert data row by row
        total_rows = len(df)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            # Prepare values
            values = []
            for val in row:
                if pd.isna(val):
                    values.append(None)
                elif isinstance(val, (datetime, pd.Timestamp)):
                    values.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    values.append(val)

            # Execute insert
            cursor.execute(insert_sql, values)

            # Show progress for large datasets
            if total_rows > 100 and i % 10 == 0:
                print(f"   Inserted {i}/{total_rows} rows...")

        # Commit transaction
        connection.commit()

        print(f"✅ SUCCESS: {total_rows} rows inserted into '{table_name}'")
        return True

    except Exception as e:
        if connection:
            connection.rollback()
        print(f"❌ ERROR: {e}")
        return False

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# -------------------- Async HTTP client --------------------
async def fetch_klines(session, symbol, interval, start_time, limit):
    """Fetch kline data asynchronously."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit
    }
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Binance API error {resp.status}: {text}")
        data = await resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'from', 'open', 'high', 'low', 'close', 'volume',
            'to', 'volume$', 'order_number', 'active_volume',
            'active_volume$', 'unknown'
        ])
        # Convert timestamps and types
        df['from'] = (df['from'] / 1000 + 3600 * 0).astype(int)
        df['to'] = (df['to'] / 1000 + 3600 * 0).astype(int)
        for col in ['open', 'high', 'low', 'close', 'volume', 'volume$',
                    'active_volume', 'active_volume$']:
            df[col] = df[col].astype(float)
        df.insert(0, 'snapshotTime', pd.to_datetime(df['from'].values + 3600 * 8, unit='s'))
        df.insert(1, 'timestamp', df['from'].values)
        df.insert(7, 'to_snapshotTime', pd.to_datetime(df['to'].values, unit='s'))
        return df

# -------------------- Main async logic --------------------
async def update_loop():
    # Global variables
    global df4_temp, df4_old, train, raw_table, buy_table, sell_table 

    

    last_spread = 0

    raw_table = "BTCUSD_5s"
    buy_table = "BTCUSD_17280_BUY720_336_5s"
    sell_table = "BTCUSD_17280_SELL720_336_5s"
    
    # Load initial data
    query_short = f"""
        SELECT * FROM (
            SELECT * FROM {raw_table}
            ORDER BY timestamp DESC
            LIMIT 100
        ) AS recent_data
        ORDER BY timestamp ASC;
    """
    query_long = f"""
        SELECT * FROM (
            SELECT * FROM {raw_table}
            ORDER BY timestamp DESC
            LIMIT 30000
        ) AS recent_data
        ORDER BY timestamp ASC;
    """
    # Run DB fetches in thread pool (they are synchronous)
    loop = asyncio.get_running_loop()

    # Initial data loading with retry
    while True:
        try:
            df4_temp = await loop.run_in_executor(None, fetch_data_from_database, query_short)
            print(df4_temp['timestamp'].diff(1)[1:].value_counts())
            print(datetime.fromtimestamp(round(time.time(), 0)), end=" ")
            df4_old = await loop.run_in_executor(None, fetch_data_from_database, query_long)
            print(df4_old['timestamp'].diff(1)[1:].value_counts())
            break
        except Exception as e:
            print(e)
            await asyncio.sleep(1)

    # Load training data
    while True:
        try:
            query_buy_train = f"""
            SELECT DISTINCT * FROM (
                SELECT DISTINCT * FROM {buy_table}
                ORDER BY open_timestamp DESC
                LIMIT 100
            ) AS last_rows
            ORDER BY open_timestamp ASC;
            """
            train_buy = await loop.run_in_executor(None, fetch_data_from_database, query_buy_train)
            print(train_buy['open_timestamp'].diff(1)[1:].value_counts())
            break
        except Exception as e:
            print(e)
            await asyncio.sleep(1)

    while True:
        try:
            query_sell_train = f"""
            SELECT DISTINCT * FROM (
                SELECT DISTINCT * FROM {sell_table}
                ORDER BY open_timestamp DESC
                LIMIT 100
            ) AS last_rows
            ORDER BY open_timestamp ASC;
            """
            train_sell = await loop.run_in_executor(None, fetch_data_from_database, query_sell_train)
            print(train_sell['open_timestamp'].diff(1)[1:].value_counts())
            break
        except Exception as e:
            print(e)
            await asyncio.sleep(1)

            
    symbol = "BTCUSDT"
    table = "BTCUSD_5s"
    database = "BTCUSD"
    db_config = mysql_config(cursorclass=DictCursor, connect_timeout=2, autocommit=True)
    # Semaphore to limit concurrent requests (Binance allows ~1200 weight per minute)
    sem = asyncio.Semaphore(5)

    async def fetch_with_semaphore(session, interval, start, limit):
        async with sem:
            return await fetch_klines(session, symbol, interval, start, limit)

    async with aiohttp.ClientSession() as session:
        while True:
            # Wait until the next minute's 3-second mark (e.g., 00:01:03, 00:02:03, etc.)
            start = time.time()
           
            # await asyncio.sleep(sleep_sec)
            interval = "1s"

            limit = 1000
            last_ts = int(df4_temp['timestamp'].iloc[-1])
            # start_date = datetime(2026, 4, 10, 0, 0, 0)
            # last_ts = int(start_date.timestamp())
            candles_per_loop = 1000    # Binance max limit per request
            total_seconds = int(time.time() - last_ts)  # total seconds from start to now
            total_loop = total_seconds // candles_per_loop + 1
        
            df_insert = pd.DataFrame()
            insert_threshold = 1
            data_frames_1m = []
            # for i in tqdm(range(total_loop)):
            for i in range(total_loop):   
                start_ms = int((last_ts + i * 1000) * 1000)
                df_1s = await fetch_with_semaphore(session, interval, start_ms, limit)
                
                if len(df_1s) > 0:
                    df_5s = df_reshape_1s(5, df_1s)
                    
            
                
                if df_5s is not None and len(df_5s) > 0:
                    # Append to accumulator
                    df_insert = pd.concat([df_insert.copy(), df_5s.copy()], ignore_index=True)
                    df_insert = df_insert.drop_duplicates(subset='timestamp')
                    df_insert.reset_index(drop=True, inplace=True)
        
                    if len(df_insert) >= insert_threshold:
                        print(f"\nAccumulated {len(df_insert)} rows. Inserting into database...")
                        print(df_insert.tail(1)[["snapshotTime","close"]])
                        data_frames_1m.append(df_insert)
                        
                        success = append_to_db(df_insert, table_name=raw_table, db_config=db_config)
                        df4_temp = df_insert.copy()
                        if success:
                            # Clear the accumulator after successful insert
                            df_insert = pd.DataFrame()
                        else:
                            print("Insertion failed, stopping.")
                            break


            if data_frames_1m:
                df_insert = pd.concat(data_frames_1m, ignore_index=True)
                df_insert = df_insert.drop_duplicates(subset='timestamp')
                df_insert = df_insert[df_insert['timestamp'] > last_ts]
                
            # Process training data generation
            df4_old = pd.concat([df4_old.copy(), df_insert.copy()], ignore_index=True)
            df4_old = df4_old.drop_duplicates(subset=['timestamp'])
            df4_old = df4_old.tail(25000).copy()
            df4_old.index = np.arange(len(df4_old))
            
            # df8 = pd.concat([df4_old.copy(), df_insert.copy()], ignore_index=True)
            # df8 = df8.drop_duplicates(subset=['timestamp'])
            # df8.index = np.arange(len(df8))

            # print(df8['snapshotTime'].values[-1], df8['timestamp'].diff(1)[1:].value_counts())

            # df8.open = df8.open.astype(float)
            # df8.close = df8.close.astype(float)
            # df8.high = df8.high.astype(float)
            # df8.low = df8.low.astype(float)
            # df8.volume = df8.volume.astype(float)

            # print(df8['timestamp'].diff(1)[1:].value_counts())

            buy_last_open = train_buy.open_timestamp.iloc[-1]
            sell_last_open = train_sell.open_timestamp.iloc[-1] 
            
            # current_spread = len(df8.query(f"timestamp > {buy_last_open}"))
            # print(last_spread, current_spread)
            # last_spread = current_spread

            # # Generate training features
            total_df = df4_old.copy()
            close_series = total_df['close']

            periods = [12]
            current = 12
            step = 12  # First step from 5 to 30
            
            total = int(3600 * 24 / 5)
            target = 12 * 60
            
            
            while current < total:
                current += step
            
                if current <= total:
                    periods.append(current)

                if current >= target:
                    step = 60
                    
                if current >= total:
                    step = 60

            # Create shifts for all periods
            shifts = pd.DataFrame({
                f'shift_{period}': close_series.shift(period)
                for period in periods
            })

            # Calculate all pct_changes at once
            changes = np.round((close_series.values[:, None] - shifts.values) / shifts.values * 100, 4)

            # Convert to DataFrame with proper column names
            changes_df = pd.DataFrame(
                changes,
                columns=[f'change{period}' for period in periods],
                index=total_df.index
            )

            total_df = pd.concat([total_df, changes_df], axis=1)
            total_df = total_df.dropna()

            total_df1 = total_df[total_df['timestamp'] % 5 == 0]
            total_df1.index = np.arange(len(total_df1))

            shifts_count = int(target / 1)
            # total_df1.insert(7, "change", total_df1[f"change{target}"].shift(-shifts_count) * 100)
            # total_df1.insert(8, "buy_sell", "buy")
            # total_df1.insert(9, "holding_time", 3600 * 1)
            # total_df1.insert(10, "moving", 3)
            # total_df1 = total_df1.drop(['high', 'low', 'open', 'volume'], axis=1)
            # total_df1.insert(6, "open_timestamp", total_df1.timestamp)
            # total_df1.insert(7, "close_timestamp", total_df1.timestamp.shift(-int(shifts_count)))
            # total_df1.insert(8, "open_date", total_df1.snapshotTime)
            # total_df1.insert(9, "close_date", total_df1.snapshotTime.shift(-int(shifts_count)))
            # total_df1["holding_time"] = total_df1["close_timestamp"] - total_df1["open_timestamp"]

            total_df1.insert(4,"change",total_df1[f"change{target}"].shift(-shifts_count)*100)
            total_df1.insert(5,"buy_sell","buy")
            total_df1.insert(6,"holding_time",3600 * 1)
            
            # total_df1.insert(10, "moving", 3)
            total_df1 = total_df1.drop(['high', 'low','open','open'], axis=1)
            
            total_df1.insert(6, "open_timestamp",total_df1.timestamp)
            total_df1.insert(7, "close_timestamp",total_df1.timestamp.shift(-int(shifts_count)))
            total_df1.insert(8, "open_date",total_df1.snapshotTime)
            total_df1.insert(9, "close_date",total_df1.snapshotTime.shift(-int(shifts_count)))
            total_df1["holding_time"] = total_df1["close_timestamp"] - total_df1["open_timestamp"]

            total_df1 = total_df1.dropna()

            train_temp_iloc = total_df1.query(f"open_timestamp > {buy_last_open}").copy()
            train_temp_iloc.index = np.arange(len(train_temp_iloc))

            # Insert training data
            train_temp_iloc = total_df1.query(f"open_timestamp > {buy_last_open}").copy()
            train_temp_iloc.index = np.arange(len(train_temp_iloc))
            
            # train_temp_iloc_filtered = train_temp_iloc.copy().drop(columns=['change1'])
            
            if not train_temp_iloc.empty:
                
                success = dataframe_to_mysql_direct(
                    df=train_temp_iloc,
                    table_name=buy_table,
                    if_exists="append"
                )
                # success = append_to_db(train_temp_iloc, table_name="BTCUSD_17280_BUY720_288_5s", db_config=db_config)

                # success = dataframe_to_mysql_direct(
                #     df=train_temp_iloc_filtered,
                #     table_name="BTCUSD_4320buy60_1min",
                #     if_exists="append"
                # )
                
                while True:
                    try:
                        query_buy_train = f"""
                        SELECT DISTINCT * FROM (
                            SELECT DISTINCT * FROM {buy_table}
                            ORDER BY open_timestamp DESC
                            LIMIT 10
                        ) AS last_rows
                        ORDER BY open_timestamp ASC;
                        """
                        train_buy = await loop.run_in_executor(None, fetch_data_from_database, query_buy_train)
                        print('buy_last_open:', train_buy['open_date'].values[-1])
                        break
                    except Exception as e:
                        print(e)
                        await asyncio.sleep(1)

                
            total_df1['change'] = -total_df1['change']
            total_df1['buy_sell'] = "sell"

      
            train_temp_iloc = total_df1.query(f"open_timestamp > {sell_last_open}").copy()
            train_temp_iloc.index = np.arange(len(train_temp_iloc))

            # train_temp_iloc_filtered = train_temp_iloc.drop(columns=['change1']).copy()
            
            if not train_temp_iloc.empty:
                success = dataframe_to_mysql_direct(
                    df=train_temp_iloc,
                    table_name=sell_table,
                    if_exists="append"
                )
                # success = append_to_db(train_temp_iloc, table_name="BTCUSD_17280_SELL720_288_5s", db_config=db_config)

                # success = dataframe_to_mysql_direct(
                #     df=train_temp_iloc_filtered,
                #     table_name="BTCUSD_4320sell60_1min",
                #     if_exists="append"
                # )
                
                while True:
                    try:
                        query_sell_train = f"""
                        SELECT DISTINCT * FROM (
                            SELECT DISTINCT * FROM {sell_table}
                            ORDER BY open_timestamp DESC
                            LIMIT 10
                        ) AS last_rows
                        ORDER BY open_timestamp ASC;
                        """
                        train_sell = await loop.run_in_executor(None, fetch_data_from_database, query_sell_train)
                        print('sell_last_open:', train_sell['open_date'].values[-1])
                        break
                    except Exception as e:
                        print(e)
                        await asyncio.sleep(1)

            end = time.time()
            sleep_sec = 5.5 - (start % 5)
            cost = round(end-start,1)
            print("cost:",cost)
            if cost < sleep_sec:
                sleep_sec = sleep_sec - cost
                await asyncio.sleep(sleep_sec)

# -------------------- Entry point --------------------
if __name__ == "__main__":
    asyncio.run(update_loop())