"""
Backfill a gap in btcusd_5s raw data using Binance 1m klines.

Current gap (detected 2026-05-04):
  555s between 1777824250 and 1777824805
  = 2026-05-03 16:04:10 → 16:13:25 HKT
  = 111 missing 5s rows

Strategy: fetch 1m klines for the gap period from Binance, resample each
1m candle into 12 × 5s bars via linear interpolation, then INSERT with
ON CONFLICT DO NOTHING into btcusd_5s.
"""
import asyncio
import aiohttp
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from db_env import db_config

DB_CONFIG = db_config(min_size=1, max_size=2)

RAW_TABLE = "btcusd_5s"
SYMBOL = "BTCUSDT"


async def fetch_1m_klines(session, start_ts, end_ts):
    """Fetch 1-minute klines covering [start_ts, end_ts] (unix epoch seconds)."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_ts * 1000
    end_ms = end_ts * 1000

    while current_start < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": "1m",
            "startTime": int(current_start),
            "limit": 500,
        }
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Binance API error {resp.status}: {text}")
            data = await resp.json()
            if not data:
                break
            all_klines.extend(data)
            # Move to next batch (last kline open time + 1 minute)
            current_start = data[-1][0] + 60000

    return all_klines


def resample_1m_to_5s(klines):
    """Resample 1-minute klines to 5-second bars via linear interpolation."""
    rows = []
    for k in klines:
        ts_open = k[0] // 1000  # unix seconds (UTC)
        o = float(k[1])
        h = float(k[2])
        l = float(k[3])
        c = float(k[4])
        v = float(k[5])

        # Generate 12 × 5s bars within this 1m window
        for i in range(12):
            bar_ts = ts_open + i * 5
            frac = (i + 1) / 12  # progress through the minute
            # Interpolate close linearly across the minute
            bar_close = o + (c - o) * frac
            bar_open = o + (c - o) * (i / 12) if i > 0 else o
            rows.append({
                "timestamp": bar_ts,
                "open": round(bar_open, 2),
                "high": h,
                "low": l,
                "close": round(bar_close, 2),
                "volume": round(v / 12, 8),
            })
    return pd.DataFrame(rows)


async def backfill_gap(pool, gap_start, gap_end):
    """Backfill a gap in btcusd_5s using 1m Binance data."""
    # Add 1-minute padding to ensure full coverage
    fetch_start = gap_start - 60
    fetch_end = gap_end + 60

    async with aiohttp.ClientSession() as session:
        print(f"Fetching 1m klines: {datetime.fromtimestamp(fetch_start, tz=timezone.utc)} → "
              f"{datetime.fromtimestamp(fetch_end, tz=timezone.utc)}")
        klines = await fetch_1m_klines(session, fetch_start, fetch_end)
    print(f"Got {len(klines)} 1m klines")

    if not klines:
        print("No klines returned, cannot backfill.")
        return

    df = resample_1m_to_5s(klines)

    # Clip to gap range (strict boundaries)
    df = df[(df['timestamp'] >= gap_start) & (df['timestamp'] <= gap_end)].copy()
    print(f"Generated {len(df)} × 5s bars for range [{gap_start}, {gap_end}]")

    if df.empty:
        print("No 5s bars generated.")
        return

    # Insert into DB
    records = []
    for _, row in df.iterrows():
        dt_utc = datetime.fromtimestamp(int(row['timestamp']), tz=timezone.utc)
        records.append((
            dt_utc,         # "timestamp"
            dt_utc,         # "snapshotTime"
            row['open'], row['high'], row['low'], row['close'], row['volume'],
        ))

    async with pool.acquire() as conn:
        await conn.executemany(
            f"""
            INSERT INTO {RAW_TABLE} ("timestamp", "snapshotTime", open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT ("timestamp") DO NOTHING
            """,
            records
        )
    print(f"Inserted {len(records)} rows into {RAW_TABLE}")


async def verify():
    """Check that diffs are now all 5s in the most recent rows."""
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT ts FROM (
                    SELECT extract(epoch from "timestamp")::bigint AS ts
                    FROM {RAW_TABLE} ORDER BY "timestamp" DESC LIMIT 30000
                ) sub ORDER BY ts ASC
            """)
        timestamps = [r['ts'] for r in rows]
        diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        from collections import Counter
        counts = Counter(diffs)
        print(f"\nVerification (last {len(timestamps)} rows):")
        for d, cnt in sorted(counts.items()):
            marker = " ← GAP" if d != 5 else ""
            print(f"  diff={d:>6}  cnt={cnt}{marker}")
        non_5 = {d: cnt for d, cnt in counts.items() if d != 5}
        if non_5:
            print(f"\n⚠ {sum(non_5.values())} non-5 gaps remaining")
        else:
            print("\n✓ ALL diffs are 5s!")
    finally:
        await pool.close()


async def main():
    # The 555s gap detected from price_newDB.py output
    gap_start = 1777824250
    gap_end = 1777824805

    print(f"Gap: {gap_start} → {gap_end}  ({(gap_end - gap_start)} seconds)")
    print(f"  Start: {datetime.fromtimestamp(gap_start, tz=timezone.utc)} UTC")
    print(f"  End:   {datetime.fromtimestamp(gap_end, tz=timezone.utc)} UTC")

    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        await backfill_gap(pool, gap_start, gap_end)
    finally:
        await pool.close()

    await verify()


if __name__ == "__main__":
    asyncio.run(main())
