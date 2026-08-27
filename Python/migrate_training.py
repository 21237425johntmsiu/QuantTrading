"""Migrate training tables from MySQL to TimescaleDB — memory efficient streaming."""
import pymysql
from pymysql.cursors import SSCursor
import psycopg2
from psycopg2.extras import execute_values
import re

from db_env import db_config, mysql_config

MYSQL = mysql_config()
PG = db_config()

BUY_TABLE = "btcusd_17280_BUY720_336_5s"
SELL_TABLE = "btcusd_17280_SELL720_336_5s"
MYSQL_BUY = "BTCUSD_17280_BUY720_336_5s"
MYSQL_SELL = "BTCUSD_17280_SELL720_336_5s"
BATCH_SIZE = 500


def get_mysql_schema(table):
    """Get MySQL column names and types."""
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
                elif 'TEXT' in mysql_type or 'VARCHAR' in mysql_type or 'CHAR' in mysql_type:
                    pg_type = 'TEXT'
                else:
                    pg_type = 'TEXT'
                cols.append((name, pg_type))
            return cols
    finally:
        conn.close()


def create_pg_table(pg_conn, table_name, columns):
    """Create PostgreSQL table."""
    with pg_conn.cursor() as c:
        col_defs = [f'"{col}" {dtype}' for col, dtype in columns]
        c.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')
        c.execute(f'CREATE TABLE {table_name} ({", ".join(col_defs)})')
    pg_conn.commit()


def migrate_table(mysql_name, pg_name):
    """Stream data from MySQL to PostgreSQL row by row."""
    columns = get_mysql_schema(mysql_name)
    col_names = [c[0] for c in columns]
    print(f"Table {mysql_name}: {len(columns)} columns, streaming...")

    pg_conn = psycopg2.connect(**PG)
    try:
        create_pg_table(pg_conn, pg_name, columns)

        mysql_conn = pymysql.connect(**MYSQL, cursorclass=SSCursor)
        try:
            with mysql_conn.cursor() as c:
                c.execute(f"SELECT * FROM {mysql_name} ORDER BY open_timestamp ASC")
                total = 0
                batch = []

                while True:
                    row = c.fetchone()
                    if row is None:
                        break

                    # Convert MySQL datetime to string for psycopg2
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
                            execute_values(cur, f'INSERT INTO {pg_name} ({cols}) VALUES %s', batch, page_size=BATCH_SIZE)
                        pg_conn.commit()
                        batch = []
                        print(f"  {mysql_name}: {total} rows", end='\r')

                # Final batch
                if batch:
                    cols = ', '.join(f'"{c}"' for c in col_names)
                    with pg_conn.cursor() as cur:
                        execute_values(cur, f'INSERT INTO {pg_name} ({cols}) VALUES %s', batch, page_size=BATCH_SIZE)
                    pg_conn.commit()

                print(f"\n  {mysql_name}: ✅ {total} rows migrated")
        finally:
            mysql_conn.close()
    finally:
        pg_conn.close()


if __name__ == "__main__":
    migrate_table(MYSQL_BUY, BUY_TABLE)
    migrate_table(MYSQL_SELL, SELL_TABLE)
