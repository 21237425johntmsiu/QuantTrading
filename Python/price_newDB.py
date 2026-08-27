import asyncio
import aiohttp
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import asyncpg

# ──────────────────── Configuration ────────────────────
from db_env import db_config

DB_CONFIG = db_config(min_size=2, max_size=10)

RAW_TABLE = "btcusd_5s"
BUY_TABLE = "btcusd_17280_BUY720_336_5s"
SELL_TABLE = "btcusd_17280_SELL720_336_5s"
BUY_SIGNAL_TABLE = "btcusd_17280_BUY720_336_5s_singal"
SELL_SIGNAL_TABLE = "btcusd_17280_SELL720_336_5s_singal"
SYMBOL = "BTCUSDT"
SEMAPHORE_LIMIT = 5


# ──────────────────── Schema Init ────────────────────
async def init_db():
    """Create connection pool, hypertable, indexes, continuous aggregates."""
    pool = await asyncpg.create_pool(**DB_CONFIG)

    try:
        async with pool.acquire() as conn:
            # Ensure TimescaleDB extension is installed
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # Raw 5s hypertable
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS btcusd_5s (
                    "timestamp" TIMESTAMPTZ NOT NULL,
                    "snapshotTime" TIMESTAMPTZ,
                    open FLOAT8 NOT NULL,
                    high FLOAT8 NOT NULL,
                    low FLOAT8 NOT NULL,
                    close FLOAT8 NOT NULL,
                    volume FLOAT8 NOT NULL
                );
            """)

            # Add snapshotTime if upgrading existing table
            await conn.execute("""
                ALTER TABLE btcusd_5s ADD COLUMN IF NOT EXISTS "snapshotTime" TIMESTAMPTZ;
            """)

            # Convert to hypertable if not already
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM _timescaledb_catalog.hypertable WHERE table_name = 'btcusd_5s')"
            )
            if not exists:
                await conn.execute(
                    "SELECT create_hypertable('btcusd_5s', 'timestamp', chunk_time_interval => INTERVAL '1 day');"
                )
                print("Created hypertable btcusd_5s")

            # Unique index for ON CONFLICT DO NOTHING
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_btcusd_5s_ts ON btcusd_5s ("timestamp");
            """)

            # Compression for chunks older than 7 days
            await conn.execute("ALTER TABLE btcusd_5s SET (timescaledb.compress);")
            try:
                await conn.execute(
                    "SELECT add_compression_policy('btcusd_5s', INTERVAL '7 days', if_not_exists => TRUE);"
                )
            except Exception as e:
                print(f"Warning: could not set compression policy: {e}")

            # Signal tables for buy/sell predictions
            for signal_table in [BUY_SIGNAL_TABLE, SELL_SIGNAL_TABLE]:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {signal_table} (
                        symbol TEXT,
                        close FLOAT8,
                        check_buy FLOAT8,
                        check_buy2 FLOAT8,
                        check_buy3 FLOAT8 DEFAULT 0,
                        open_timestamp BIGINT
                    );
                """)
                # Add columns if table already existed without them
                for col, col_type in [("check_buy3", "FLOAT8 DEFAULT 0"), ("close", "FLOAT8")]:
                    try:
                        await conn.execute(f"""
                            ALTER TABLE {signal_table} ADD COLUMN IF NOT EXISTS {col} {col_type};
                        """)
                    except Exception:
                        pass

        return pool
    except Exception:
        await pool.close()
        raise


# ──────────────────── Binance API ────────────────────
async def fetch_klines(session, symbol, interval, start_time, limit):
    """Fetch kline data from Binance asynchronously."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start_time, "limit": limit}
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
        df['from'] = (df['from'] / 1000).astype(int)
        df['to'] = (df['to'] / 1000).astype(int)
        for col in ['open', 'high', 'low', 'close', 'volume', 'volume$',
                     'active_volume', 'active_volume$']:
            df[col] = df[col].astype(float)
        df.insert(0, 'snapshotTime', pd.to_datetime(df['from'], unit='s') + pd.Timedelta(hours=8))
        df.insert(1, 'timestamp', df['from'].values)
        df.insert(7, 'to_snapshotTime', pd.to_datetime(df['to'].values, unit='s'))
        return df


# ──────────────────── Resampling ────────────────────
def df_reshape_1s(length, df1):
    """Resample 1-second data to N-second OHLCV."""
    df = df1.copy()
    if 'timestamp' not in df.columns and 'snapshotTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['snapshotTime']).apply(lambda x: int(x.timestamp()))
    else:
        df['timestamp'] = df['timestamp'].astype('int64')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df_resampled = df.resample(f'{length}s').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'datetime': 'snapshotTime'}, inplace=True)
    df_resampled['from_timestamp'] = df_resampled['snapshotTime'].apply(lambda x: int(x.timestamp()))
    df_resampled['to_timestamp'] = df_resampled['from_timestamp'] + length
    df_resampled['snapshotTime'] = df_resampled['snapshotTime'] + pd.Timedelta(hours=8)
    df_resampled.rename(columns={'from_timestamp': 'timestamp'}, inplace=True)
    return df_resampled[["snapshotTime", "timestamp", "open", "close", "high", "low", "volume"]]


# ──────────────────── Feature Engineering ────────────────────
# compute_features / prepare_train_data live in ../feature_method.py
# (private, outside Python/) — importable via db_env's sys.path hook.
from feature_method import compute_features_24h as compute_features, prepare_train_data


# ──────────────────── DB Helpers ────────────────────
async def insert_5s_batch(pool, df):
    """Insert 5s OHLCV rows into hypertable. Returns count of rows processed."""
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        ts = int(row['timestamp'])
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        records.append((
            dt_utc,
            dt_utc,
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']),
        ))

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO btcusd_5s ("timestamp", "snapshotTime", open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT ("timestamp") DO NOTHING
            """,
            records
        )
    return len(records)


async def create_dynamic_table(conn, df, table_name):
    """Create table with columns matching DataFrame dtypes."""
    col_defs = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'int' in dtype:
            col_type = 'BIGINT'
        elif 'float' in dtype:
            col_type = 'FLOAT8'
        elif 'datetime' in dtype or 'timestamp' in dtype:
            col_type = 'TIMESTAMPTZ' if df[col].dt.tz is not None else 'TIMESTAMP'
        else:
            col_type = 'TEXT'
        col_defs.append(f'"{col}" {col_type}')
    await conn.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(col_defs)});')
    # Ensure unique index on open_timestamp for ON CONFLICT support
    if 'open_timestamp' in df.columns:
        try:
            await conn.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS uq_{table_name}_open_ts ON {table_name} (open_timestamp);'
            )
        except Exception:
            pass


async def insert_train_batch(pool, df, table_name):
    """Insert training data into buy/sell table, creating dynamic columns if needed."""
    if df.empty:
        return
    columns = list(df.columns)

    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
            table_name
        )
        if not exists:
            await create_dynamic_table(conn, df, table_name)

        records = []
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                elif isinstance(val, pd.Timestamp):
                    row_data.append(val.to_pydatetime())
                else:
                    row_data.append(val)
            records.append(tuple(row_data))

        cols = ', '.join(f'"{c}"' for c in columns)
        placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
        await conn.executemany(
            f'INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT (open_timestamp) DO NOTHING',
            records
        )


async def reload_train_data(pool, table_name, order_col="open_timestamp", limit=10):
    """Fetch latest N rows from a training table."""
    query = f"""
        SELECT * FROM (
            SELECT * FROM {table_name}
            ORDER BY {order_col} DESC
            LIMIT {limit}
        ) AS sub
        ORDER BY {order_col} ASC;
    """
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(query)
        except Exception as e:
            print(f"Warning: could not query {table_name}: {e}")
            return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df


async def load_initial_data(pool, query, max_retries=30):
    """Load data from DB into DataFrame with retry."""
    for attempt in range(max_retries):
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame([dict(r) for r in rows])
            # safety: catch any unquoted lowercased column aliases
            if 'snapshottime' in df.columns:
                df = df.rename(columns={'snapshottime': 'snapshotTime'})
            # Convert timezone-aware timestamps to naive HKT for consistency
            if pd.api.types.is_datetime64_any_dtype(df['snapshotTime']) and df['snapshotTime'].dt.tz is not None:
                df['snapshotTime'] = df['snapshotTime'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            return df
        except Exception as e:
            print(f"DB load error (attempt {attempt+1}/{max_retries}): {e}, retrying...")
            await asyncio.sleep(1)
    print(f"Failed to load data after {max_retries} attempts")
    return pd.DataFrame()


# ──────────────────── Raw Data Gap Backfill ────────────────────
async def backfill_raw_gap(pool, session, gap_start, gap_end):
    """Backfill a single gap in btcusd_5s using 1s klines reshaped to 5s."""
    limit = 1000
    all_5s = []

    current_start = gap_start
    while current_start <= gap_end:
        start_ms = int(current_start * 1000)
        try:
            df_1s = await fetch_klines(session, SYMBOL, "1s", start_ms, limit)
        except Exception as e:
            print(f"  Binance 1s fetch error at {current_start}: {e}")
            current_start += limit
            continue

        if df_1s.empty:
            current_start += limit
            continue

        df_5s = df_reshape_1s(5, df_1s)
        if not df_5s.empty:
            all_5s.append(df_5s)
        current_start += limit

    if not all_5s:
        print(f"  No 1s data available for gap {gap_start}→{gap_end} (may be too old)")
        return 0

    df_insert = pd.concat(all_5s, ignore_index=True)
    df_insert = df_insert.drop_duplicates(subset='timestamp')
    df_insert = df_insert[(df_insert['timestamp'] >= gap_start) & (df_insert['timestamp'] <= gap_end)]

    if df_insert.empty:
        return 0

    inserted = await insert_5s_batch(pool, df_insert)
    print(f"  Backfilled {inserted} rows for gap {gap_start}→{gap_end}")
    return inserted


async def detect_and_backfill_gaps(pool, df):
    """Detect gaps in raw data and backfill from Binance 1s klines reshaped to 5s."""
    if df.empty or len(df) < 10:
        return

    diffs = df['timestamp'].diff()
    gaps = []
    for i in range(1, len(diffs)):
        d = diffs.iloc[i]
        if d != 5:
            gap_start = int(df['timestamp'].iloc[i - 1]) + 5
            gap_end = int(df['timestamp'].iloc[i]) - 5
            gaps.append((gap_start, gap_end))

    if not gaps:
        return

    total_missing = sum(ge - gs + 5 for gs, ge in gaps)
    print(f"Detected {len(gaps)} gap(s) in raw data ({total_missing}s total):")
    for gs, ge in gaps:
        print(f"  {gs} → {ge}  ({(ge - gs + 5)}s, {(ge - gs + 5) // 5} rows)  "
              f"[{datetime.fromtimestamp(gs, tz=timezone.utc)} → {datetime.fromtimestamp(ge, tz=timezone.utc)}]")

    async with aiohttp.ClientSession() as session:
        for gs, ge in gaps:
            await backfill_raw_gap(pool, session, gs, ge)


# ──────────────────── Main Loop ────────────────────
async def update_loop():
    pool = await init_db()

    # ── Load initial raw data ──
    query_short = f"""
        SELECT "snapshotTime",
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {RAW_TABLE} ORDER BY "timestamp" DESC LIMIT 100
        ) AS sub ORDER BY "timestamp" ASC;
    """
    query_long = f"""
        SELECT "snapshotTime",
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {RAW_TABLE} ORDER BY "timestamp" DESC LIMIT 30000
        ) AS sub ORDER BY "timestamp" ASC;
    """

    df4_temp = await load_initial_data(pool, query_short)
    if not df4_temp.empty:
        print("Temp diff counts:")
        print(df4_temp['timestamp'].diff(1)[1:].value_counts())
    else:
        print("No historical data found. Will start fresh from Binance.")
        now_ts = int(time.time())
        df4_temp = pd.DataFrame([{
            'snapshotTime': datetime.fromtimestamp(now_ts - 5, tz=timezone.utc),
            'timestamp': now_ts - 5,
            'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0
        }])

    df4_old = await load_initial_data(pool, query_long)
    if not df4_old.empty:
        print("Old diff counts:")
        print(df4_old['timestamp'].diff(1)[1:].value_counts())

    # ── Backfill any gaps in raw data ──
    if not df4_old.empty and len(df4_old) > 10:
        await detect_and_backfill_gaps(pool, df4_old)
        # Reload both temp and old after backfill
        df4_temp = await load_initial_data(pool, query_short)
        df4_old = await load_initial_data(pool, query_long)
        if not df4_old.empty:
            print("After backfill diff counts:")
            print(df4_old['timestamp'].diff(1)[1:].value_counts())

    # ── Load training data ──
    train_buy = await reload_train_data(pool, BUY_TABLE)
    train_sell = await reload_train_data(pool, SELL_TABLE)

    # ── Binance fetch semaphore ──
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def fetch_with_semaphore(session, interval, start, limit):
        async with sem:
            return await fetch_klines(session, SYMBOL, interval, start, limit)

    # ── Main loop ──
    async with aiohttp.ClientSession() as session:
        try:
            while True:
                start = time.time()
                interval = "1s"
                limit = 1000

                last_ts = int(df4_temp['timestamp'].iloc[-1])
                total_seconds = max(0, int(time.time() - last_ts))
                total_loop = total_seconds // limit + 1

                df_insert = pd.DataFrame()

                for i in range(total_loop):
                    start_ms = int((last_ts + i * limit) * 1000)
                    df_1s = await fetch_with_semaphore(session, interval, start_ms, limit)

                    if df_1s.empty:
                        continue

                    df_5s = df_reshape_1s(5, df_1s)
                    if not df_5s.empty:
                        df_5s = df_5s.drop_duplicates(subset='timestamp')
                        df_insert = pd.concat([df_insert, df_5s], ignore_index=True)

                # Insert all accumulated 5s data to DB
                if not df_insert.empty:
                    await insert_5s_batch(pool, df_insert)
                    last = df_insert.iloc[-1]
                    last_time = last['snapshotTime']
                    print(f"5s row: {last_time} O={last['open']} H={last['high']} L={last['low']} C={last['close']} V={last['volume']:.4f}")

                # Update in-memory state
                if not df_insert.empty:
                    df4_temp = df_insert.copy()
                    # Concatenate; column name mismatch (snapshottime vs snapshotTime) is
                    # already gone because load_initial_data renames snapshottime.
                    df4_old = pd.concat([df4_old, df_insert], ignore_index=True)
                    df4_old = df4_old.drop_duplicates(subset=['timestamp'])
                    df4_old = df4_old.tail(25000).copy()
                    df4_old.index = np.arange(len(df4_old))

                if df4_old.empty:
                    end = time.time()
                    cost = round(end - start, 1)
                    sleep_sec = 5.5 - (start % 5)
                    print(f"cost: {cost}")
                    if cost < sleep_sec:
                        await asyncio.sleep(sleep_sec - cost)
                    continue

                # ── Feature engineering ──
                total_df, periods, target = compute_features(df4_old)
                shifts_count = target

                # ── Buy training data ──
                buy_last_open = train_buy['open_timestamp'].iloc[-1] if not train_buy.empty else 0
                buy_df = prepare_train_data(total_df, target, shifts_count, "buy")
                buy_new = buy_df[buy_df['open_timestamp'] > buy_last_open].copy()
                if not buy_new.empty:
                    buy_new.index = np.arange(len(buy_new))
                    await insert_train_batch(pool, buy_new, BUY_TABLE)
                    print(f"inserted buy: {len(buy_new)} rows")
                    train_buy = await reload_train_data(pool, BUY_TABLE)
                    if not train_buy.empty:
                        print(f"buy_last_open: {train_buy['open_date'].iloc[-1]}")

                # ── Sell training data ──
                sell_last_open = train_sell['open_timestamp'].iloc[-1] if not train_sell.empty else 0
                sell_df = prepare_train_data(total_df, target, shifts_count, "sell")
                sell_df['change'] = -sell_df['change']
                sell_new = sell_df[sell_df['open_timestamp'] > sell_last_open].copy()
                if not sell_new.empty:
                    sell_new.index = np.arange(len(sell_new))
                    await insert_train_batch(pool, sell_new, SELL_TABLE)
                    print(f"inserted sell: {len(sell_new)} rows")
                    train_sell = await reload_train_data(pool, SELL_TABLE)
                    if not train_sell.empty:
                        print(f"sell_last_open: {train_sell['open_date'].iloc[-1]}")

                # ── Sleep until next 5s boundary ──
                end = time.time()
                sleep_sec = 5.5 - (start % 5)
                cost = round(end - start, 1)
                print(f"cost: {cost}")
                if cost < sleep_sec:
                    await asyncio.sleep(sleep_sec - cost)
        finally:
            await pool.close()


# ──────────────────── Entry Point ────────────────────
if __name__ == "__main__":
    asyncio.run(update_loop())
