#!/usr/bin/env python3
"""
BTCUSD_all_in_one_MT5.py
Trade execution counterpart to BTCUSD_all_in_one.py.
Reads signals from btcusd_120960_signal table and executes trades via MetaTrader5.
"""

import asyncio
import asyncpg
import time
from datetime import datetime
import datetime as raw_datetime

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    print("[MT5 not available — trading functions disabled]")

# ──────────────────── Configuration ────────────────────
from db_env import db_config, mt5_credentials

DB_CONFIG = db_config(min_size=2, max_size=10)

SIGNAL_TABLE = "btcusd_120960_signal"
symbol = 'BTCUSD'
order_size = 0.01
MAX_HOLDING_TIME = 3600
MIN_TIME_BETWEEN_ORDERS = 300

# MT5 credentials (from root env.txt)
_MT5 = mt5_credentials()
MT5_LOGIN = _MT5["login"]
MT5_PASSWORD = _MT5["password"]
MT5_SERVER = _MT5["server"]


# ──────────────────── MT5 Functions ────────────────────

def connect_to_mt5():
    if mt5 is None:
        print("MT5 not available, skipping connect_to_mt5")
        return False
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False
    authorized = mt5.login(
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER
    )
    if authorized:
        print(f"Connected to account #{MT5_LOGIN}, server: {MT5_SERVER}")
        return True
    else:
        print("Login failed, error code =", mt5.last_error())
        return False


def buy_order(symbol, volume):
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


# ──────────────────── Market Check ────────────────────

def stop_market():
    return True


# ──────────────────── Async DB ────────────────────

async def fetch_latest_signal(pool, side_val, after_ts=0):
    """Fetch the latest non-zero signal for a given side."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"""
            SELECT * FROM {SIGNAL_TABLE}
            WHERE side = $1 AND open_timestamp > $2 AND singal1 != 0
            ORDER BY open_timestamp DESC
            LIMIT 1
        """, side_val, after_ts)
        return row


async def fetch_last_signal_row(pool):
    """Fetch the most recent row from the signal table (for display)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"""
            SELECT * FROM {SIGNAL_TABLE}
            ORDER BY open_timestamp DESC
            LIMIT 1
        """)
        return row


# ──────────────────── Main Loop ────────────────────

async def main_loop():
    pool = await asyncpg.create_pool(**DB_CONFIG)

    # Per-side order tracking
    buy_orders = {"order_ID": [], "start_time": []}
    sell_orders = {"order_ID": [], "start_time": []}

    last_processed_buy_ts = 0
    last_processed_sell_ts = 0

    # ── MT5 Connection & Position Recovery ──
    if connect_to_mt5():
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                if "BUY5" in pos.comment:
                    buy_orders["order_ID"].append(pos.ticket)
                    buy_orders["start_time"].append(pos.time + 3600 * 5)
                elif "SELL5" in pos.comment:
                    sell_orders["order_ID"].append(pos.ticket)
                    sell_orders["start_time"].append(pos.time + 3600 * 5)

    print(f"buy_orders: {buy_orders}")
    print(f"sell_orders: {sell_orders}")

    try:
        while True:
            start = time.time()

            # ── Display latest signal row ──
            now_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            last_row = await fetch_last_signal_row(pool)
            if last_row:
                print(f"[{now_dt}] Last signal row:")
                for key in last_row.keys():
                    print(f"  {key}: {last_row[key]}")
            else:
                print(f"[{now_dt}] Signal table is empty")

            # ── Check buy signal ──
            buy_signal = await fetch_latest_signal(pool, 'buy', last_processed_buy_ts)
            if buy_signal is not None and stop_market():
                singal1 = float(buy_signal['singal1'])
                singal2 = float(buy_signal['singal2'])
                last_processed_buy_ts = int(buy_signal['open_timestamp'])

                print(f"Buy signal check: {singal1} >= 0.70, {singal2} >= 0.70")

                if singal1 >= 0.70 and singal2 >= 0.70:
                    current_time = time.time()
                    recent_buy = False
                    if buy_orders["start_time"]:
                        latest_buy = max(buy_orders["start_time"])
                        time_since = current_time - latest_buy + 3600 * 8
                        if time_since < MIN_TIME_BETWEEN_ORDERS:
                            print(f"Skip buy: only {time_since:.1f}s since last order, "
                                  f"need {MIN_TIME_BETWEEN_ORDERS}s")
                            recent_buy = True

                    if not recent_buy:
                        trade_try = 3
                        while trade_try > 0:
                            trade_try -= 1
                            try:
                                if not connect_to_mt5():
                                    print(f"MT5 connect failed, {trade_try} retries left")
                                    continue
                                trade_result = buy_order(symbol, order_size)
                                if trade_result is not None:
                                    oid = trade_result.order
                                    print(f"Buy trade opened! Order ID: {oid}")
                                    buy_orders["order_ID"].append(oid)
                                    buy_orders["start_time"].append(time.time() + 3600 * 8)
                                    break
                                else:
                                    print(f"Buy trade returned None, {trade_try} retries left")
                            except AttributeError as e:
                                print(e)
                        else:
                            print(f"Failed to open buy trade after retries")

            # ── Check sell signal ──
            sell_signal = await fetch_latest_signal(pool, 'sell', last_processed_sell_ts)
            if sell_signal is not None and stop_market():
                singal1 = float(sell_signal['singal1'])
                singal2 = float(sell_signal['singal2'])
                last_processed_sell_ts = int(sell_signal['open_timestamp'])

                print(f"Sell signal check: {singal1} >= 0.21, {singal2} >= 0.41")

                if singal1 >= 0.21 and singal2 >= 0.41:
                    current_time = time.time()
                    recent_sell = False
                    if sell_orders["start_time"]:
                        latest_sell = max(sell_orders["start_time"])
                        time_since = current_time - latest_sell + 3600 * 8
                        if time_since < MIN_TIME_BETWEEN_ORDERS:
                            print(f"Skip sell: only {time_since:.1f}s since last order, "
                                  f"need {MIN_TIME_BETWEEN_ORDERS}s")
                            recent_sell = True

                    if not recent_sell:
                        trade_try = 3
                        while trade_try > 0:
                            trade_try -= 1
                            try:
                                if not connect_to_mt5():
                                    print(f"MT5 connect failed, {trade_try} retries left")
                                    continue
                                trade_result = sell_order(symbol, order_size)
                                if trade_result is not None:
                                    oid = trade_result.order
                                    print(f"Sell trade opened! Order ID: {oid}")
                                    sell_orders["order_ID"].append(oid)
                                    sell_orders["start_time"].append(time.time() + 3600 * 8)
                                    break
                                else:
                                    print(f"Sell trade returned None, {trade_try} retries left")
                            except AttributeError as e:
                                print(e)
                        else:
                            print(f"Failed to open sell trade after retries")

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
                if "BUY5" in position.comment:
                    holding_time = time.time() - position.time + 3600 * 3
                    if holding_time >= MAX_HOLDING_TIME:
                        close_position_by_ticket(ticket=position.ticket)
                elif "SELL5" in position.comment:
                    holding_time = time.time() - position.time + 3600 * 3
                    if holding_time >= MAX_HOLDING_TIME:
                        close_position_by_ticket(ticket=position.ticket)

            # ── Sleep until next 5s boundary ──
            now = time.time()
            cost = round((now - start), 2)
            sleep_sec = 6 - (start % 5)

            if cost > sleep_sec:
                print('cost:', cost, "sleep_sec:", 0)
            else:
                real_sleep = round((sleep_sec - cost), 2)
                print('cost:', cost, "sleep_sec:", real_sleep)
                await asyncio.sleep(real_sleep)

            print()
    finally:
        await pool.close()


# ──────────────────── Entry Point ────────────────────

if __name__ == "__main__":
    asyncio.run(main_loop())
