"""
Rename column `time` -> `timestamp` in btcusd_5s hypertable.
Also updates dependent objects: continuous aggregate, indexes, policies.
"""
import asyncio
import asyncpg

from db_env import db_config

DB_CONFIG = db_config()

RAW_TABLE = "btcusd_5s"
AGG_VIEW = "btcusd_1m"


async def migration():
    pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            # ---- Step 0: Check if already migrated ----
            col_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name=$1 AND column_name=$2)",
                RAW_TABLE, "timestamp"
            )
            if col_exists:
                print("Column 'timestamp' already exists in btcusd_5s. Checking if 'time' is gone...")
                time_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_name=$1 AND column_name=$2)",
                    RAW_TABLE, "time"
                )
                if not time_exists:
                    print("Migration already complete (time column gone, timestamp exists).")
                    return

            # ---- Step 1: Drop continuous aggregate and its policy ----
            print("1. Dropping continuous aggregate policy for btcusd_1m...")
            try:
                await conn.execute("""
                    CALL remove_continuous_aggregate_policy('btcusd_1m');
                """)
            except Exception as e:
                print(f"   (non-fatal) {e}")

            print("2. Dropping continuous aggregate view btcusd_1m...")
            try:
                await conn.execute("DROP MATERIALIZED VIEW IF EXISTS btcusd_1m CASCADE;")
            except Exception as e:
                print(f"   (non-fatal) {e}")

            # ---- Step 2: Drop compression policy ----
            print("3. Removing compression policy...")
            try:
                await conn.execute("""
                    CALL remove_compression_policy('btcusd_5s');
                """)
            except Exception as e:
                print(f"   (non-fatal) {e}")

            # ---- Step 3: Drop indexes that reference `time` ----
            print("4. Dropping index idx_btcusd_5s_time...")
            await conn.execute("DROP INDEX IF EXISTS idx_btcusd_5s_time;")

            # ---- Step 4: Turn off compression temporarily ----
            print("5. Disabling compression...")
            try:
                await conn.execute("ALTER TABLE btcusd_5s SET (timescaledb.compress = false);")
            except Exception as e:
                print(f"   Could not disable compression directly: {e}")
                print("   (may already be off or handled differently)")

            # ---- Step 5: Rename column ----
            print(f"6. Renaming column 'time' -> 'timestamp' in {RAW_TABLE}...")
            try:
                await conn.execute(f'ALTER TABLE {RAW_TABLE} RENAME COLUMN "time" TO "timestamp";')
                print("   Rename successful!")
            except Exception as e:
                print(f"   ERROR: {e}")
                print("   Rename failed. Rolling back dependent changes may be needed.")
                raise

            # ---- Step 6: Re-enable compression ----
            print("7. Re-enabling compression...")
            try:
                await conn.execute(f"ALTER TABLE {RAW_TABLE} SET (timescaledb.compress);")
                # Use INTERVAL '7 days' like price_newDB.py does
                await conn.execute(
                    "SELECT add_compression_policy($1, INTERVAL '7 days', if_not_exists => TRUE);",
                    RAW_TABLE
                )
                print("   Compression re-enabled.")
            except Exception as e:
                print(f"   (non-fatal) {e}")

            # ---- Step 7: Recreate unique index ----
            print("8. Recreating unique index on timestamp...")
            await conn.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS idx_btcusd_5s_ts ON {RAW_TABLE} ("timestamp");'
            )

            # ---- Step 8: Recreate continuous aggregate ----
            print("9. Re-creating continuous aggregate btcusd_1m...")
            try:
                await conn.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS btcusd_1m
                    WITH (timescaledb.continuous) AS
                    SELECT
                        time_bucket('1 min', "timestamp") AS bucket,
                        first(open, "timestamp") AS open,
                        max(high) AS high,
                        min(low) AS low,
                        last(close, "timestamp") AS close,
                        sum(volume) AS volume
                    FROM btcusd_5s
                    GROUP BY bucket;
                """)
                await conn.execute("""
                    SELECT add_continuous_aggregate_policy('btcusd_1m',
                        start_offset => INTERVAL '1 day',
                        end_offset => INTERVAL '1 min',
                        schedule_interval => INTERVAL '1 minute');
                """)
                print("   Continuous aggregate recreated.")
            except Exception as e:
                print(f"   WARNING: could not recreate continuous aggregate: {e}")
                print("   The table itself is fine; the 1m aggregate needs manual attention.")

            # ---- Step 9: Verify ----
            cols = await conn.fetch(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name=$1 ORDER BY ordinal_position",
                RAW_TABLE
            )
            print(f"\nMigration complete. Current columns in {RAW_TABLE}:")
            for c in cols:
                print(f"   {c['column_name']:20} {c['data_type']}")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(migration())
