import pandas as pd
import numpy as np
import time
from datetime import datetime
import datetime as raw_datetime

import psycopg2
import psycopg2.extras
from collections import namedtuple

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    print("[MT5 not available — trading functions disabled]")

# ─── Configuration ───
from db_env import db_config

DB_CONFIG = db_config()

table_symbol = "btcusd"
BUY_SIGNAL_TABLE = f"{table_symbol}_17280_BUY720_336_5s_singal"
SELL_SIGNAL_TABLE = f"{table_symbol}_17280_SELL720_336_5s_singal"

# ─── DB Functions ───

def fetch_data_from_database(query, db_config=None, return_type='dataframe'):
    if db_config is None:
        db_config = DB_CONFIG
    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query)
        if return_type == 'raw':
            return cursor
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        if 'open_timestamp' in df.columns:
            df['open_timestamp'] = df['open_timestamp'].astype('int64')
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        if connection:
            connection.close()


def stop_market():
    return True

    year, month, day, hour, minute = time.localtime(time.time())[:5]
    current_date = raw_datetime.date(year, month, day)
    if current_date.weekday() >= 5 and hour == 4:
        return False
    else:
        return True


# ─── MT5 Functions ───

def buy_order(symbol, volume):
    global mt5
    if mt5 is None:
        print("MT5 not available, skipping buy_order")
        return None
    try:
        if not mt5.symbol_select(symbol, True):
            print(f"Add {symbol} to Market Watch first!")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if not tick or not tick.ask:
            print("Failed to get price")
            return None
        current_price = tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": current_price,
            "deviation": 20,
            "magic": int(time.time()),
            "comment": "BUY5",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            print(f"""
            GOLD TRADE OPENED:
            Ticket: {result.order}
            Entry: {current_price:.2f}
            """)
            return result
        else:
            print(f"Failed to open trade: {mt5.last_error()}")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def sell_order(symbol, volume):
    global mt5
    if mt5 is None:
        print("MT5 not available, skipping sell_order")
        return None
    try:
        if not mt5.symbol_select(symbol, True):
            print(f"Add {symbol} to Market Watch first!")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if not tick or not tick.bid:
            print("Failed to get price")
            return None
        current_price = tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL,
            "price": current_price,
            "deviation": 20,
            "magic": int(time.time()),
            "comment": "SELL5",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            print(f"""
            GOLD SELL TRADE OPENED:
            Ticket: {result.order}
            Entry: {current_price:.2f}
            """)
            return result
        else:
            print(f"Failed to open SELL trade: {mt5.last_error()}")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def connect_to_mt5():
    global mt5
    if mt5 is None:
        print("MT5 not available, skipping connect_to_mt5")
        return False
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False
    authorized = mt5.login(
        login=login,
        password=password,
        server=server
    )
    if authorized:
        print(f"Connected to account #{login}, server: {server}")
        return True
    else:
        print("Login failed, error code =", mt5.last_error())
        return False


def close_position_by_ticket(ticket, deviation=10):
    if mt5 is None:
        return False
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"No position found with ticket #{ticket}")
            return False
        position = position[0]
        symbol_info = mt5.symbol_info(position.symbol)
        if not symbol_info:
            print(f"Failed to get market data for {position.symbol}")
            return False
        if position.type == mt5.ORDER_TYPE_BUY:
            price = symbol_info.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = symbol_info.ask
            order_type = mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": position.magic,
            "comment": f"Closed by Python script",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Successfully closed position #{ticket}", end=" ")
            print(f"Symbol: {position.symbol}", end=" ")
            print(f"Type: {'BUY' if position.type == 0 else 'SELL'}", end=" ")
            print(f"Volume: {position.volume}", end=" ")
            print(f"Close Price: {price}", end=" ")
            print(f"Profit: {position.profit}", end=" ")
            print(f"Close Time: {datetime.fromtimestamp(position.time - 3600 * 7)}")
            return True
        else:
            print(f"Failed to close position #{ticket}")
            print(f"Error code: {result.retcode}")
            print(f"Error description: {mt5.last_error()}")
            return False
    except Exception as e:
        print(f"Error closing position: {str(e)}")
        return False


# ─── Global State ───

is_buy = True
signal_table = BUY_SIGNAL_TABLE if is_buy else SELL_SIGNAL_TABLE

order_list = {
    "order_ID": [],
    "start_time": [],
}

MAX_HOLDING_TIME = 3600
MIN_TIME_BETWEEN_ORDERS = 300
symbol = 'BTCUSD'
order_size = 0.01

last_processed_ts = 0

# ─── MT5 Connection & Position Recovery ───

if connect_to_mt5():
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            expected = "BUY5" if is_buy else "SELL5"
            if expected in pos.comment:
                order_list["order_ID"].append(pos.ticket)
                order_list["start_time"].append(pos.time + 3600 * 5)

print(order_list)

# ─── Main Trading Loop ───

while True:
    start_time = time.time()

    # Display last row of signal table
    now_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_row_query = f"""
        SELECT * FROM {signal_table}
        ORDER BY open_timestamp DESC
        LIMIT 1
    """
    last_row_df = fetch_data_from_database(last_row_query)
    if not last_row_df.empty:
        print(f"[{now_dt}] Last signal row:")
        for col in last_row_df.columns:
            print(f"  {col}: {last_row_df[col].values[0]}")
    else:
        print(f"[{now_dt}] Signal table is empty")

    # Query latest signal from DB
    query = f"""
        SELECT * FROM {signal_table}
        WHERE open_timestamp > {last_processed_ts}
        ORDER BY open_timestamp DESC
        LIMIT 1
    """
    signal_df = fetch_data_from_database(query)

    if not signal_df.empty and stop_market():
        check_val = float(signal_df['check_buy'].values[0])
        check_val2 = float(signal_df['check_buy2'].values[0])
        signal_ts = int(signal_df['open_timestamp'].values[0])
        last_processed_ts = signal_ts

        threshold = 0.70 if is_buy else 0.21
        threshold2 = 0.70 if is_buy else 0.41

        print(f"Signal check: {check_val} >= {threshold}, {check_val2} >= {threshold2}")

        if check_val >= threshold and check_val2 >= threshold2:
            current_time = time.time()
            recent_order_found = False
            if order_list["start_time"]:
                latest_order_time = max(order_list["start_time"])
                time_since_last_order = current_time - latest_order_time + 3600 * 8
                if time_since_last_order < MIN_TIME_BETWEEN_ORDERS:
                    print(f"Skip order: only {time_since_last_order:.1f}s since last order, "
                          f"need {MIN_TIME_BETWEEN_ORDERS}s")
                    recent_order_found = True

            if not recent_order_found:
                trade_try = 3
                while trade_try > 0:
                    trade_try -= 1
                    try:
                        if not connect_to_mt5():
                            print(f"MT5 connect failed, {trade_try} retries left")
                            continue
                        if is_buy:
                            trade_result = buy_order(symbol, order_size)
                        else:
                            trade_result = sell_order(symbol, order_size)

                        if trade_result is not None:
                            oid = trade_result.order
                            print(f"Trade opened! Order ID: {oid}")
                            order_list["order_ID"].append(oid)
                            order_list["start_time"].append(time.time() + 3600 * 8)
                            print(f"Order recorded - OID: {oid}")
                            break
                        else:
                            print(f"Trade execution returned None, {trade_try} retries left")
                    except AttributeError as e:
                        print(e)
                else:
                    print(f"Failed to open trade after retries")

    # ── Position management ──
    positions = []
    if mt5 is not None:
        try:
            positions = mt5.positions_get()
        except Exception as e:
            print(f"MT5 positions_get error: {e}")
    if positions is None:
        positions = []

    for position in positions:
        expected = "BUY5" if is_buy else "SELL5"
        if expected in position.comment:
            holding_time = time.time() - position.time + 3600 * 3
            if holding_time >= MAX_HOLDING_TIME:
                close_position_by_ticket(ticket=position.ticket)

    # ── Sleep ──
    now = time.time()
    cost = round((now - start_time), 2)
    sleep_sec = 6 - (start_time % 5)

    if cost > sleep_sec:
        print('cost:', cost, "sleep_sec:", 0)
        pass
    else:
        real_sleep = round((sleep_sec - cost), 2)
        print('cost:', cost, "sleep_sec:", real_sleep)
        time.sleep(real_sleep)

    print()
