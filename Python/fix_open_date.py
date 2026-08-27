"""Fix open_date/close_date in training tables: update UTC+0 values to UTC+8."""
import asyncio
import asyncpg

from db_env import db_config

DB_CONFIG = db_config()

TABLES = ["btcusd_17280_BUY720_336_5s", "btcusd_17280_SELL720_336_5s"]

FIX_SQL = """
    UPDATE {table}
    SET
        open_date = to_timestamp(open_timestamp) + interval '8 hours',
        close_date = to_timestamp(close_timestamp) + interval '8 hours'
    WHERE open_date IS DISTINCT FROM (to_timestamp(open_timestamp) + interval '8 hours')
       OR close_date IS DISTINCT FROM (to_timestamp(close_timestamp) + interval '8 hours');
"""


async def main():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        async with pool.acquire() as conn:
            for table in TABLES:
                # Count mismatches before fix
                before = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE open_date IS DISTINCT FROM
                        (to_timestamp(open_timestamp) + interval '8 hours')::timestamp without time zone
                       OR close_date IS DISTINCT FROM
                        (to_timestamp(close_timestamp) + interval '8 hours')::timestamp without time zone;
                """)
                total = await conn.fetchval(f"SELECT COUNT(*) FROM {table};")

                # Apply fix
                updated = await conn.execute(FIX_SQL.format(table=table))
                # asyncpg returns "UPDATE X" string
                updated_count = int(updated.split()[-1])

                print(f"{table}: {before} mismatched out of {total} total → {updated_count} rows updated")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
