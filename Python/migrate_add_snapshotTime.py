"""
Add snapshotTime TIMESTAMPTZ column to btcusd_5s, backfilled from "timestamp".
"""
import asyncio
import asyncpg

from db_env import db_config

DB_CONFIG = db_config()

RAW_TABLE = "btcusd_5s"


async def migrate():
    pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            # Allow decompression of more than 100k tuples for backfill
            await conn.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0;")

            # Check if column already exists
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name=$1 AND column_name=$2)",
                RAW_TABLE, "snapshotTime"
            )
            if exists:
                print("snapshotTime column already exists, skipping add.")
                return

            print("Adding snapshotTime TIMESTAMPTZ column...")
            await conn.execute(f'ALTER TABLE {RAW_TABLE} ADD COLUMN "snapshotTime" TIMESTAMPTZ;')

            cnt = await conn.fetchval(f'SELECT COUNT(*) FROM {RAW_TABLE} WHERE "snapshotTime" IS NULL')
            print(f"Backfilling {cnt} rows from \"timestamp\" column...")

            await conn.execute(f'UPDATE {RAW_TABLE} SET "snapshotTime" = "timestamp" WHERE "snapshotTime" IS NULL;')

            # Verify
            nulls = await conn.fetchval(f'SELECT COUNT(*) FROM {RAW_TABLE} WHERE "snapshotTime" IS NULL')
            sample = await conn.fetch(f'SELECT "timestamp", "snapshotTime" FROM {RAW_TABLE} LIMIT 3')
            print(f"Remaining nulls: {nulls}")
            for r in sample:
                print(f"  ts={r['timestamp']}  snap={r['snapshotTime']}")

            print("Migration complete.")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(migrate())
