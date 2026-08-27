"""Migrate sell training table from MySQL to TimescaleDB."""
import pymysql
from pymysql.cursors import SSCursor
import psycopg2
from psycopg2.extras import execute_values

from db_env import db_config, mysql_config

MYSQL = mysql_config()
PG = db_config()

SELL_TABLE = "btcusd_17280_SELL720_336_5s"
MYSQL_SELL = "BTCUSD_17280_SELL720_336_5s"
BATCH_SIZE = 500


def get_mysql_schema(table):
    conn = pymysql.connect(**MYSQL, cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as c:
            c.execute(f"SHOW COLUMNS FROM {table}")
            cols = []
            for row in c.fetchall():
                name = row['Field']
                mysql_type = row['Type'].upper()
                if 'INT' in mysql_type:
                    pg_type = 'BIGINT'
                elif 'DOUBLE' in mysql_type or 'FLOAT' in mysql_type or 'DECIMAL' in mysql_type:
                    pg_type = 'FLOAT8'
                elif 'DATETIME' in mysql_type or 'TIMESTAMP' in mysql_type:
                    pg_type = 'TIMESTAMPTZ'
                else:
                    pg_type = 'TEXT'
                cols.append((name, pg_type))
            return cols
    finally:
        conn.close()


def migrate_sell():
    columns = get_mysql_schema(MYSQL_SELL)
    col_names = [c[0] for c in columns]
    print(f"Sell table: {len(columns)} columns, streaming...")

    pg_conn = psycopg2.connect(**PG)
    try:
        # Drop and recreate
        with pg_conn.cursor() as c:
            c.execute(f'DROP TABLE IF EXISTS {SELL_TABLE} CASCADE')
            col_defs = [f'"{col}" {dtype}' for col, dtype in columns]
            c.execute(f'CREATE TABLE {SELL_TABLE} ({", ".join(col_defs)})')
        pg_conn.commit()

        mysql_conn = pymysql.connect(**MYSQL, cursorclass=SSCursor)
        try:
            with mysql_conn.cursor() as c:
                c.execute(f"SELECT * FROM {MYSQL_SELL} ORDER BY open_timestamp ASC")
                total = 0
                batch = []

                while True:
                    row = c.fetchone()
                    if row is None:
                        break
                    processed = []
                    for val in row:
                        if hasattr(val, 'strftime'):
                            processed.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                        else:
                            processed.append(val)
                    batch.append(processed)
                    total += 1

                    if len(batch) >= BATCH_SIZE:
                        cols = ', '.join(f'"{c}"' for c in col_names)
                        with pg_conn.cursor() as cur:
                            execute_values(cur, f'INSERT INTO {SELL_TABLE} ({cols}) VALUES %s', batch, page_size=BATCH_SIZE)
                        pg_conn.commit()
                        batch = []
                        print(f"  Sell: {total} rows", end='\r')

                if batch:
                    cols = ', '.join(f'"{c}"' for c in col_names)
                    with pg_conn.cursor() as cur:
                        execute_values(cur, f'INSERT INTO {SELL_TABLE} ({cols}) VALUES %s', batch, page_size=BATCH_SIZE)
                    pg_conn.commit()

                print(f"\n  Sell: ✅ {total} rows migrated")
        finally:
            mysql_conn.close()
    finally:
        pg_conn.close()


if __name__ == "__main__":
    migrate_sell()
