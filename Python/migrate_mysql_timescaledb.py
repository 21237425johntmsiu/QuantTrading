"""Migrate BTCUSD data from MySQL (192.168.1.86) to local TimescaleDB."""
import asyncio
import asyncpg
import pandas as pd
import pymysql
from datetime import timezone
from pymysql.cursors import DictCursor

# ── Config (credentials from root env.txt) ──
from db_env import db_config, mysql_config

MYSQL_CONFIG = mysql_config(cursorclass=DictCursor)

PG_CONFIG = db_config()

RAW_TABLE = "btcusd_5s"
BUY_TABLE = "btcusd_17280_BUY720_336_5s"
SELL_TABLE = "btcusd_17280_SELL720_336_5s"
CHUNK_SIZE = 10000


# ── MySQL Helpers ──
def fetch_mysql_chunks(query, chunk_size=CHUNK_SIZE):
    """Generator yielding DataFrames from MySQL in chunks."""
    conn = pymysql.connect(**MYSQL_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows)
    finally:
        conn.close()


# ── TimescaleDB Helpers ──
async def init_pg(pool):
    """Ensure hypertable and training tables exist."""
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS btcusd_5s (
                time TIMESTAMPTZ NOT NULL,
                open FLOAT8 NOT NULL,
                high FLOAT8 NOT NULL,
                low FLOAT8 NOT NULL,
                close FLOAT8 NOT NULL,
                volume FLOAT8 NOT NULL
            );
        """)
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM _timescaledb_catalog.hypertable WHERE table_name = 'btcusd_5s')"
        )
        if not exists:
            await conn.execute(
                "SELECT create_hypertable('btcusd_5s', 'time', chunk_time_interval => INTERVAL '1 day');"
            )
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_btcusd_5s_time ON btcusd_5s (time);
        """)


async def create_dynamic_table(conn, df, table_name):
    """Create table matching DataFrame columns."""
    col_defs = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'int' in dtype:
            col_type = 'BIGINT'
        elif 'float' in dtype:
            col_type = 'FLOAT8'
        elif 'datetime' in dtype or 'timestamp' in dtype:
            col_type = 'TIMESTAMPTZ'
        else:
            col_type = 'TEXT'
        col_defs.append(f'"{col}" {col_type}')
    await conn.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(col_defs)});')


async def insert_raw_chunk(pool, df):
    """Insert a chunk of 5s raw data into TimescaleDB hypertable."""
    if df.empty:
        return
    records = []
    for _, row in df.iterrows():
        records.append((
            row['timestamp'],          # Unix epoch from MySQL
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']),
        ))
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO btcusd_5s (time, open, high, low, close, volume)
               VALUES (to_timestamp($1), $2, $3, $4, $5, $6)
               ON CONFLICT (time) DO NOTHING""",
            records
        )


async def insert_train_chunk(pool, df, table_name):
    """Insert a chunk of training data."""
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
            f'INSERT INTO {table_name} ({cols}) VALUES ({placeholders})',
            records
        )


# ── Main Migration ──
async def migrate():
    pool = await asyncpg.create_pool(**PG_CONFIG)
    try:
        await init_pg(pool)
        total_raw = 0

        # ── 1. Migrate BTCUSD_5s → btcusd_5s ──
        print("Migrating BTCUSD_5s → btcusd_5s...")
        query = "SELECT timestamp, open, high, low, close, volume FROM BTCUSD_5s ORDER BY timestamp ASC"
        for i, chunk in enumerate(fetch_mysql_chunks(query)):
            await insert_raw_chunk(pool, chunk)
            total_raw += len(chunk)
            print(f"  Raw chunk {i+1}: {len(chunk)} rows (total: {total_raw})")
        print(f"✅ Raw data migrated: {total_raw} rows")

        # ── 2. Migrate buy training table ──
        print(f"\nMigrating BTCUSD_17280_BUY720_336_5s → {BUY_TABLE}...")
        query_buy = "SELECT * FROM BTCUSD_17280_BUY720_336_5s ORDER BY open_timestamp ASC"
        total_buy = 0
        for i, chunk in enumerate(fetch_mysql_chunks(query_buy)):
            await insert_train_chunk(pool, chunk, BUY_TABLE)
            total_buy += len(chunk)
            print(f"  Buy chunk {i+1}: {len(chunk)} rows (total: {total_buy})")
        print(f"✅ Buy training data migrated: {total_buy} rows")

        # ── 3. Migrate sell training table ──
        print(f"\nMigrating BTCUSD_17280_SELL720_336_5s → {SELL_TABLE}...")
        query_sell = "SELECT * FROM BTCUSD_17280_SELL720_336_5s ORDER BY open_timestamp ASC"
        total_sell = 0
        for i, chunk in enumerate(fetch_mysql_chunks(query_sell)):
            await insert_train_chunk(pool, chunk, SELL_TABLE)
            total_sell += len(chunk)
            print(f"  Sell chunk {i+1}: {len(chunk)} rows (total: {total_sell})")
        print(f"✅ Sell training data migrated: {total_sell} rows")

        # ── Verify ──
        async with pool.acquire() as conn:
            raw_cnt = await conn.fetchval("SELECT count(*) FROM btcusd_5s")
            buy_cnt = await conn.fetchval(f"SELECT count(*) FROM {BUY_TABLE}")
            sell_cnt = await conn.fetchval(f"SELECT count(*) FROM {SELL_TABLE}")
            t_range = await conn.fetchval("SELECT min(time) FROM btcusd_5s")
            t_range2 = await conn.fetchval("SELECT max(time) FROM btcusd_5s")
            print(f"\n{'='*50}")
            print(f"Verification:")
            print(f"  btcusd_5s:                  {raw_cnt} rows")
            print(f"  {BUY_TABLE}:  {buy_cnt} rows")
            print(f"  {SELL_TABLE}: {sell_cnt} rows")
            print(f"  Date range: {t_range} → {t_range2}")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(migrate())
