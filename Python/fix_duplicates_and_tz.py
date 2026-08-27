"""Fix duplicates and convert open_date/close_date to naive UTC+8 TIMESTAMP.

Steps:
  1. Delete duplicate rows (keep one per open_timestamp)
  2. ALTER open_date/close_date from TIMESTAMPTZ → TIMESTAMP
  3. Recompute open_date/close_date as correct UTC+8 naive values from timestamps
"""
import asyncio
import asyncpg

from db_env import db_config

DB_CONFIG = db_config()

TABLES = ["btcusd_17280_BUY720_336_5s", "btcusd_17280_SELL720_336_5s"]


async def fix_table(pool, table):
    print(f"\n=== {table} ===")

    async with pool.acquire() as conn:
        # --- Step 1: Count state ---
        total_before = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

        # --- Step 2: Deduplicate — keep one row per open_timestamp ---
        await conn.execute(f"""
            DELETE FROM {table}
            WHERE ctid IN (
                SELECT ctid FROM (
                    SELECT ctid, open_timestamp,
                           ROW_NUMBER() OVER (
                               PARTITION BY open_timestamp
                               ORDER BY ABS(EXTRACT(EPOCH FROM open_date) - open_timestamp)
                           ) AS rn
                    FROM {table}
                ) sub
                WHERE rn > 1
            )
        """)
        after_dedup = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
        removed_dupes = total_before - after_dedup
        print(f"  Rows before dedup:  {total_before}")
        print(f"  Duplicates removed: {removed_dupes}")
        print(f"  After dedup:        {after_dedup}")

        # --- Step 3: ALTER column types to TIMESTAMP ---
        # Drop default first if any, then alter
        for col in ['open_date', 'close_date']:
            await conn.execute(f"""
                ALTER TABLE {table} ALTER COLUMN {col} TYPE timestamp
                USING {col}::timestamp
            """)
        print(f"  ✓ Columns altered to TIMESTAMP")

        # --- Step 4: Recompute open_date/close_date from timestamps as UTC+8 naive ---
        await conn.execute(f"""
            UPDATE {table} SET
                open_date = (to_timestamp(open_timestamp) + interval '8 hours')::timestamp,
                close_date = (to_timestamp(close_timestamp::bigint) + interval '8 hours')::timestamp
        """)
        print(f"  ✓ Dates recomputed from timestamps")

        # --- Step 5: Verify ---
        total_after = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

        # No rows should have open_date differing from expected UTC+8 by more than 1s
        bad = await conn.fetchval(f"""
            SELECT COUNT(*) FROM {table}
            WHERE ABS(
                EXTRACT(EPOCH FROM open_date) - (open_timestamp + 28800)
            ) > 1
        """)

        print(f"  Final row count:    {total_after}")
        print(f"  Bad open_date:      {bad} (should be 0)")

        # Show sample
        sample = await conn.fetch(f"""
            SELECT open_timestamp, open_date, close_date
            FROM {table}
            ORDER BY open_timestamp DESC
            LIMIT 5
        """)
        print(f"  Sample rows:")
        for r in sample:
            print(f"    ts={r['open_timestamp']}  open_date={r['open_date']}  "
                  f"close_date={r['close_date']}")


async def main():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        for table in TABLES:
            await fix_table(pool, table)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
