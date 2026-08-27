"""
Backfill gaps in buy/sell training tables caused by insufficient history
during early script runs. The raw btcusd_5s table has complete data now.

Strategy:
  1. Load a wide window of raw data (24h before first gap → 1h after last gap)
     so that change17280 (24h lookback) and change720.shift(-720) (1h lookahead)
     can be computed for all gap timestamps.
  2. Run the same feature engineering as price_newDB.py.
  3. Compute training rows for the missing open_timestamp ranges only.
  4. INSERT with ON CONFLICT DO NOTHING (unique constraint on open_timestamp).
"""
import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from db_env import db_config

DB_CONFIG = db_config(min_size=1, max_size=2)

RAW_TABLE = "btcusd_5s"
BUY_TABLE = "btcusd_17280_BUY720_336_5s"
SELL_TABLE = "btcusd_17280_SELL720_336_5s"


# ─── Identified gaps (open_timestamp) ───
# From analysis:
#   Buy:  12:00:05..12:05:20 (row at 12:00:00 exists, 12:05:25 exists)
#   Buy:  14:00:05..15:03:50 (row at 14:00:00 exists, 15:03:55 exists)
#   Sell: 14:20:45..15:04:40 (row at 14:20:40 exists, 15:04:45 exists)
# These timestamps are HKT (UTC+8). The btcusd_5s snapshotTime is stored
# as TIMESTAMPTZ; we query by unix epoch.

BUY_GAPS = [
    (1777780805, 1777781120),   # 12:00:05 → 12:05:20 (65 rows)
    (1777788005, 1777791830),   # 14:00:05 → 15:03:50 (767 rows)
]
SELL_GAPS = [
    (1777789245, 1777791880),   # 14:20:45 → 15:04:40 (529 rows)
]

# Context needed for compute_features: 17280 shifts (24h) + 720 forward shift (1h)
LOOKBACK = 17280 * 5  # 86400 seconds (24h)
LOOKAHEAD = 720 * 5   # 3600 seconds (1h)


# Feature method lives in ../feature_method.py (private, outside Python/).
from feature_method import compute_features_24h as compute_features, prepare_train_data


async def main():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        # ── Build the union range of all gaps ──
        all_ranges = BUY_GAPS + SELL_GAPS
        gap_start = min(r[0] for r in all_ranges)
        gap_end = max(r[1] for r in all_ranges)

        query_start = gap_start - LOOKBACK
        query_end = gap_end + LOOKAHEAD

        print(f"Gap range (unix ts): {gap_start} → {gap_end}")
        print(f"  ({datetime.fromtimestamp(gap_start, tz=timezone.utc)} → "
              f"{datetime.fromtimestamp(gap_end, tz=timezone.utc)})")
        print(f"Query range: {query_start} → {query_end}")

        # ── Load raw data ──
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT extract(epoch from "snapshotTime")::bigint AS timestamp,
                       open, high, low, close, volume
                FROM {RAW_TABLE}
                WHERE extract(epoch from "snapshotTime")::bigint >= $1
                  AND extract(epoch from "snapshotTime")::bigint <= $2
                ORDER BY "snapshotTime" ASC
            """, query_start, query_end)

        if not rows:
            print("No raw data found in range. Is btcusd_5s populated?")
            return

        df = pd.DataFrame([dict(r) for r in rows])
        print(f"Loaded {len(df)} raw rows from btcusd_5s")

        # ── Compute features ──
        total_df, periods, target = compute_features(df)
        print(f"After compute_features: {len(total_df)} rows (dropped NaN)")

        # ── Generate buy training rows for gap timestamps ──
        shifts_count = target
        buy_df = prepare_train_data(total_df, target, shifts_count, "buy")

        for gs, ge in BUY_GAPS:
            mask = (buy_df['open_timestamp'] >= gs) & (buy_df['open_timestamp'] <= ge)
            missing = buy_df[mask]
            if missing.empty:
                print(f"  Buy gap {gs}→{ge}: no rows to insert")
                continue
            print(f"  Buy gap {gs}→{ge}: inserting {len(missing)} rows")
            # Insert into DB
            records = [tuple(row) for row in missing.to_numpy()]
            cols = ', '.join(f'"{c}"' for c in missing.columns)
            placeholders = ', '.join(f'${i+1}' for i in range(len(missing.columns)))
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'INSERT INTO {BUY_TABLE} ({cols}) VALUES ({placeholders}) ON CONFLICT (open_timestamp) DO NOTHING',
                    records
                )

        # ── Generate sell training rows for gap timestamps ──
        sell_df = prepare_train_data(total_df, target, shifts_count, "sell")
        sell_df['change'] = -sell_df['change']

        for gs, ge in SELL_GAPS:
            mask = (sell_df['open_timestamp'] >= gs) & (sell_df['open_timestamp'] <= ge)
            missing = sell_df[mask]
            if missing.empty:
                print(f"  Sell gap {gs}→{ge}: no rows to insert")
                continue
            print(f"  Sell gap {gs}→{ge}: inserting {len(missing)} rows")
            records = [tuple(row) for row in missing.to_numpy()]
            cols = ', '.join(f'"{c}"' for c in missing.columns)
            placeholders = ', '.join(f'${i+1}' for i in range(len(missing.columns)))
            async with pool.acquire() as conn:
                await conn.executemany(
                    f'INSERT INTO {SELL_TABLE} ({cols}) VALUES ({placeholders}) ON CONFLICT (open_timestamp) DO NOTHING',
                    records
                )

        # ── Verify ──
        for label, tbl in [("Buy", BUY_TABLE), ("Sell", SELL_TABLE)]:
            async with pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT diff, COUNT(*) AS cnt FROM (
                        SELECT open_timestamp - LAG(open_timestamp) OVER (ORDER BY open_timestamp) AS diff
                        FROM {tbl}
                    ) sub WHERE diff IS NOT NULL AND diff != 5
                    GROUP BY diff ORDER BY diff
                """)
            if rows:
                print(f"\n{label} table remaining non-5 diffs:")
                for r in rows:
                    print(f"  diff={r['diff']}  cnt={r['cnt']}")
            else:
                print(f"\n{label} table: ALL diffs are 5s ✓")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
