"""Verify open_timestamp is UTC+0 and open_date is UTC+8 in training tables."""
import asyncio
from datetime import datetime, timezone
import asyncpg
import pandas as pd

from db_env import db_config

DB_CONFIG = db_config()

BUY_TABLE = "btcusd_17280_BUY720_336_5s"
SELL_TABLE = "btcusd_17280_SELL720_336_5s"
LIMIT = 10

QUERY = """
    SELECT open_timestamp, open_date
    FROM {table}
    ORDER BY open_timestamp DESC
    LIMIT {limit};
"""


async def check_table(pool, table_name):
    print(f"\n{'='*60}")
    print(f"  Table: {table_name}")
    print(f"{'='*60}")

    async with pool.acquire() as conn:
        rows = await conn.fetch(QUERY.format(table=table_name, limit=LIMIT))

    if not rows:
        print("  (empty)")
        return

    print(f"  {'open_timestamp':>15}  {'open_date':>22}  {'from_ts_utc+0':>22}  {'from_ts_utc+8':>22}  {'match?':>6}")
    print(f"  {'─'*15}  {'─'*22}  {'─'*22}  {'─'*22}  {'─'*6}")

    for row in rows:
        ts = row['open_timestamp']
        od = row['open_date']

        utc0_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        utc8_str = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(
            timezone(timedelta(hours=8))
        ).strftime('%Y-%m-%d %H:%M:%S')

        if isinstance(od, str):
            od_str = od
        elif hasattr(od, 'strftime'):
            od_str = od.strftime('%Y-%m-%d %H:%M:%S')
        else:
            od_str = str(od)

        match = "✓" if utc8_str[:16] == od_str[:16] else "✗"

        print(f"  {ts:>15}  {od_str:>22}  {utc0_str:>22}  {utc8_str:>22}  {match:>6}")

    # Summary
    mismatches = 0
    for row in rows:
        ts = row['open_timestamp']
        od = row['open_date']
        if isinstance(od, str):
            od_dt = od
        elif hasattr(od, 'strftime'):
            od_dt = od.strftime('%Y-%m-%d %H:%M:%S')
        else:
            od_dt = str(od)
        utc8_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(
            timezone(timedelta(hours=8))
        ).strftime('%Y-%m-%d %H:%M:%S')
        if od_dt[:16] != utc8_dt[:16]:
            mismatches += 1

    print(f"\n  Result: {mismatches}/{len(rows)} mismatches")
    if mismatches == 0:
        print("  ✓ open_date matches open_timestamp converted to UTC+8")
    else:
        print("  ✗ Some rows have mismatched dates!")


async def main():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        await check_table(pool, BUY_TABLE)
        await check_table(pool, SELL_TABLE)
    finally:
        await pool.close()


if __name__ == "__main__":
    from datetime import timedelta
    asyncio.run(main())
