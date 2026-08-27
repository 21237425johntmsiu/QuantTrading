#!/usr/bin/env python3
"""
BTCUSD_all_in_one.py
Combined: price_newDB.py (async data pipeline) + BTCUSD_LSTM_part3_336_5s_singal.py (ML training + signals)
"""

import asyncio
import asyncpg
import time
import os
from datetime import datetime, timezone
import datetime as raw_datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

# ──────────────────── Configuration ────────────────────
from db_env import asset_path, db_config

DB_CONFIG = db_config(min_size=2, max_size=10)

RAW_TABLE = "btcusd_5s"
table_symbol = "btcusd"
total_period = 120960
BUY_TABLE = f"{table_symbol}_{total_period}_BUY720_336_5s"
SELL_TABLE = f"{table_symbol}_{total_period}_SELL720_336_5s"
SIGNAL_TABLE = f"{table_symbol}_{total_period}_signal"
TrainingResult = namedtuple('TrainingResult', ['model', 'train_losses'])
SequenceData = namedtuple('SequenceData', ['features', 'target'])


# ──────────────────── ML Classes ────────────────────

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class FinancialSequenceDataset(Dataset):
    def __init__(self, dataframe, target_column, feature_columns, seq_len=60,
                 scale_features=True, scale_target=False,
                 feature_scaler=None, target_scaler=None):
        self.dataframe = dataframe.copy()
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.seq_len = seq_len

        features_raw = self.dataframe[self.feature_columns].values
        target_raw = self.dataframe[self.target_column].values

        if scale_features:
            if feature_scaler is not None:
                self.feature_scaler = feature_scaler
                self.features = self.feature_scaler.transform(features_raw)
            else:
                self.feature_scaler = StandardScaler()
                self.features = self.feature_scaler.fit_transform(features_raw)
        else:
            self.feature_scaler = None
            self.features = features_raw

        if scale_target:
            if target_scaler is not None:
                self.target_scaler = target_scaler
                self.target = self.target_scaler.transform(target_raw.reshape(-1, 1)).flatten()
            else:
                self.target_scaler = StandardScaler()
                self.target = self.target_scaler.fit_transform(target_raw.reshape(-1, 1)).flatten()
        else:
            self.target_scaler = None
            self.target = target_raw

    def __len__(self):
        return len(self.dataframe) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.features[idx:idx + self.seq_len]
        y_target = self.target[idx + self.seq_len - 1]
        return SequenceData(
            features=torch.FloatTensor(x_seq),
            target=torch.FloatTensor([y_target])
        )


class FinancialTransformerTrainer:
    def __init__(self, model, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.model_path = model_path

    def train_multiple_epochs(self, train_loader, optimizer, num_epochs=10, save_dir='120960', save_interval=50):
        self.model.train()
        train_losses = []
        os.makedirs(save_dir, exist_ok=True)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                batch_x, batch_y = batch.features.to(self.device), batch.target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)
            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f'{self.model_path}_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, save_path)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f} - Model saved to {save_path}")
        return TrainingResult(self.model, train_losses)

    def load_model_by_epoch(self, epoch, save_dir='120960'):
        save_path = os.path.join(save_dir, f'{self.model_path}{epoch}.pth')
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint not found at {save_path}")
        checkpoint = torch.load(save_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Loaded model from epoch {checkpoint['epoch']} with loss: {checkpoint['loss']:.4f}")
        return checkpoint

    def predict(self, test_df, epoch=None, save_dir='120960',
                feature_columns_start=11, target_column='change', seq_len=1,
                feature_scaler=None, feature_columns=None):
        if epoch is not None:
            self.load_model_by_epoch(epoch, save_dir)
        self.model.eval()
        self.model.to(self.device)
        if feature_columns is None:
            feature_columns = test_df.columns[feature_columns_start:].tolist()
        test_dataset = FinancialSequenceDataset(
            test_df, target_column, feature_columns, seq_len=seq_len,
            scale_features=True, feature_scaler=feature_scaler
        )
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        with torch.no_grad():
            for sample in test_dataset:
                batch_x = sample.features.to(self.device)
                batch_y = sample.target.to(self.device)
                if seq_len == 1:
                    batch_x = batch_x.unsqueeze(0)
                    batch_y = batch_y.unsqueeze(0)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        y_pred = []
        for sublist in all_predictions:
            if isinstance(sublist, (list, np.ndarray)):
                for element in sublist:
                    y_pred.append(float(element))
            else:
                y_pred.append(float(sublist))
        y_pred_value = round(y_pred[-1], 4)
        return y_pred_value

    def predict_single(self, sample_data, epoch=None, save_dir='120960'):
        if epoch is not None:
            self.load_model_by_epoch(epoch, save_dir)
        self.model.eval()
        if isinstance(sample_data, np.ndarray):
            sample_tensor = torch.FloatTensor(sample_data)
        elif isinstance(sample_data, pd.DataFrame):
            sample_tensor = torch.FloatTensor(sample_data.values)
        else:
            sample_tensor = torch.FloatTensor(sample_data)
        if sample_tensor.dim() == 1:
            sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
        elif sample_tensor.dim() == 2:
            sample_tensor = sample_tensor.unsqueeze(0)
        sample_tensor = sample_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.model(sample_tensor)
        return prediction.cpu().item()


# ──────────────────── Schema Init ────────────────────

async def init_db():
    """Create connection pool, hypertable, indexes, continuous aggregates, signal tables."""
    pool = await asyncpg.create_pool(**DB_CONFIG)

    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS btcusd_5s (
                    "timestamp" TIMESTAMPTZ NOT NULL,
                    "snapshotTime" TIMESTAMPTZ,
                    open FLOAT8 NOT NULL,
                    high FLOAT8 NOT NULL,
                    low FLOAT8 NOT NULL,
                    close FLOAT8 NOT NULL,
                    volume FLOAT8 NOT NULL
                );
            """)

            await conn.execute("""
                ALTER TABLE btcusd_5s ADD COLUMN IF NOT EXISTS "snapshotTime" TIMESTAMPTZ;
            """)

            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM _timescaledb_catalog.hypertable WHERE table_name = 'btcusd_5s')"
            )
            if not exists:
                await conn.execute(
                    "SELECT create_hypertable('btcusd_5s', 'timestamp', chunk_time_interval => INTERVAL '1 day');"
                )
                print("Created hypertable btcusd_5s")

            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_btcusd_5s_ts ON btcusd_5s ("timestamp");
            """)

            await conn.execute("ALTER TABLE btcusd_5s SET (timescaledb.compress);")
            try:
                await conn.execute(
                    "SELECT add_compression_policy('btcusd_5s', INTERVAL '7 days', if_not_exists => TRUE);"
                )
            except Exception as e:
                print(f"Warning: could not set compression policy: {e}")

            # Signal table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {SIGNAL_TABLE} (
                    symbol TEXT,
                    side TEXT,
                    close FLOAT8,
                    "window" INT,
                    singal1 FLOAT8,
                    singal2 FLOAT8,
                    singal3 FLOAT8 DEFAULT 0,
                    open_timestamp BIGINT,
                    model_select TEXT DEFAULT NULL
                );
            """)

        return pool
    except Exception:
        await pool.close()
        raise


# ──────────────────── Period Generation & Features ────────────────────
# generate_periods / compute_features / prepare_train_data live in
# ../feature_method.py (private, outside Python/).
from feature_method import (
    generate_periods,
    compute_features_7d as compute_features,
    prepare_train_data,
    bin_change_int_series,
    bin_change15_int_series,
)


# ──────────────────── DB Helpers ────────────────────

async def create_dynamic_table(conn, df, table_name):
    """Create table with columns matching DataFrame dtypes."""
    col_defs = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'int' in dtype:
            col_type = 'BIGINT'
        elif 'float' in dtype:
            col_type = 'FLOAT8'
        elif 'datetime' in dtype or 'timestamp' in dtype:
            col_type = 'TIMESTAMPTZ' if df[col].dt.tz is not None else 'TIMESTAMP'
        else:
            col_type = 'TEXT'
        col_defs.append(f'"{col}" {col_type}')
    await conn.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(col_defs)});')
    if 'open_timestamp' in df.columns:
        try:
            await conn.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS uq_{table_name}_open_ts ON {table_name} (open_timestamp);'
            )
        except Exception:
            pass


async def insert_train_batch(pool, df, table_name):
    """Insert training data into buy/sell table, creating dynamic columns if needed."""
    if df.empty:
        return
    columns = list(df.columns)

    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
            table_name
        )
        if not exists:
            await create_dynamic_table(conn, df, table_name)

        records = []
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                elif isinstance(val, pd.Timestamp):
                    row_data.append(val.to_pydatetime())
                else:
                    row_data.append(val)
            records.append(tuple(row_data))

        cols = ', '.join(f'"{c}"' for c in columns)
        placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
        await conn.executemany(
            f'INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT (open_timestamp) DO NOTHING',
            records
        )


async def reload_train_data(pool, table_name, order_col="open_timestamp", limit=10):
    """Fetch latest N rows from a training table."""
    query = f"""
        SELECT * FROM (
            SELECT * FROM {table_name}
            ORDER BY {order_col} DESC
            LIMIT {limit}
        ) AS sub
        ORDER BY {order_col} ASC;
    """
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(query)
        except Exception as e:
            print(f"Warning: could not query {table_name}: {e}")
            return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df


async def load_initial_data(pool, query, max_retries=30):
    """Load data from DB into DataFrame with retry."""
    for attempt in range(max_retries):
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame([dict(r) for r in rows])
            if 'snapshottime' in df.columns:
                df = df.rename(columns={'snapshottime': 'snapshotTime'})
            if pd.api.types.is_datetime64_any_dtype(df['snapshotTime']) and df['snapshotTime'].dt.tz is not None:
                df['snapshotTime'] = df['snapshotTime'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            return df
        except Exception as e:
            print(f"DB load error (attempt {attempt+1}/{max_retries}): {e}, retrying...")
            await asyncio.sleep(1)
    print(f"Failed to load data after {max_retries} attempts")
    return pd.DataFrame()


async def load_train_data_full(pool, table_name, limit=10000):
    """Load training data for pre-training (up to limit rows)."""
    query = f"""
        SELECT DISTINCT * FROM (
            SELECT DISTINCT * FROM {table_name}
            ORDER BY open_timestamp DESC
            LIMIT {limit}
        ) AS last_rows
        ORDER BY open_timestamp ASC;
    """
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(query)
        except Exception as e:
            print(f"Warning: could not query {table_name}: {e}")
            return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    for col in ['open_date', 'close_date']:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    return df


async def insert_signal_async(pool, side_val, symbol_val, close_val, window_val, singal1_val, singal2_val, open_ts, singal3_val=0, model_select_val=None):
    """Insert a signal row into the signal table."""
    async with pool.acquire() as conn:
        model_str = str(model_select_val) if model_select_val is not None else None
        await conn.execute(f"""
            INSERT INTO {SIGNAL_TABLE} (symbol, side, close, "window", singal1, singal2, singal3, open_timestamp, model_select)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """, symbol_val, side_val, float(close_val), int(window_val), float(singal1_val), float(singal2_val), float(singal3_val), int(open_ts), model_str)
    await trim_signal_async(pool)


async def trim_signal_async(pool, max_rows=3600):
    """Trim old rows from the signal table, keeping the latest max_rows."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            DELETE FROM {SIGNAL_TABLE}
            WHERE open_timestamp < (
                SELECT MIN(open_timestamp) FROM (
                    SELECT open_timestamp FROM {SIGNAL_TABLE}
                    ORDER BY open_timestamp DESC
                    LIMIT {max_rows}
                ) AS latest
            );
        """)


# ──────────────────── Stop Market Check ────────────────────

def stop_market():
    return True


# ──────────────────── Train Models ────────────────────

def train_models(train_df, model_length, is_buy, columnB, columnC):
    """Train 3 LSTM models on the given training DataFrame. Returns dict of model artifacts."""
    models = {}
    train = train_df.copy()

    # Compute prediction targets
    train.insert(4, "change_int", 0, allow_duplicates=False)
    shifts15 = 12 * 15
    train.insert(5, f"Y_change15", train[f"change{shifts15}"].shift(-shifts15) * 100)
    train.insert(6, "change15_int", 0, allow_duplicates=False)

    # Trim to 2-hour boundary
    while len(train) > 0 and train.iloc[-1]['open_timestamp'] % (3600 * 2) != 0:
        train = train.iloc[:-1]

    train = train.copy()[train['open_timestamp'] % 5 == 0]

    for value in model_length:
        model = train.tail(value).copy()
        model.index = np.arange(len(model))
        models[f"model{value}"] = model

    tag_prefix = "A" if is_buy else "B"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for value in model_length:
        model = models[f"model{value}"].copy()

        # --- Model 1: predict change_int (binned change720) ---
        # Binning constants moved to ../feature_method.py (private).
        model['change_int'] = bin_change_int_series(model['change'])

        feature_columns = model.columns[14:].tolist()
        target_column = "change_int"

        print(f"[{tag_prefix}1] shape: {model.shape}, features: {len(feature_columns)}")

        train_dataset = FinancialSequenceDataset(
            model, target_column, feature_columns, seq_len=12,
            scale_features=True, scale_target=False
        )
        os.makedirs("120960", exist_ok=True)
        scaler_tag = f"{tag_prefix}1"
        np.savetxt(f"120960/scaler_{scaler_tag}_mean.csv",
                   train_dataset.feature_scaler.mean_, delimiter=",")
        np.savetxt(f"120960/scaler_{scaler_tag}_scale.csv",
                   train_dataset.feature_scaler.scale_, delimiter=",")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        lstm_model = TimeSeriesLSTM(
            input_size=len(feature_columns),
            hidden_size=len(feature_columns),
            num_layers=4
        )
        model_path60 = f"LSTM_{tag_prefix}1_{value}"
        trainer = FinancialTransformerTrainer(
            model=lstm_model, model_path=model_path60, device=device
        )
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        trainer.train_multiple_epochs(train_loader, optimizer, num_epochs=100, save_interval=100)

        models[f"train_dataset{value}"] = train_dataset
        models[f"trainer{value}"] = trainer
        models[f"model_path60_{value}"] = model_path60

        # --- Model 2: predict change15_int (binned Y_change15) ---
        # Binning constants moved to ../feature_method.py (private).
        model['change15_int'] = bin_change15_int_series(model['Y_change15'])

        feature_columns2 = columnB
        target_column2 = "change15_int"

        train_dataset2 = FinancialSequenceDataset(
            model, target_column2, feature_columns2, seq_len=12,
            scale_features=True, scale_target=False
        )
        scaler_tag2 = f"{tag_prefix}2"
        np.savetxt(f"120960/scaler_{scaler_tag2}_mean.csv",
                   train_dataset2.feature_scaler.mean_, delimiter=",")
        np.savetxt(f"120960/scaler_{scaler_tag2}_scale.csv",
                   train_dataset2.feature_scaler.scale_, delimiter=",")
        train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=False)

        lstm_model2 = TimeSeriesLSTM(
            input_size=len(feature_columns2),
            hidden_size=len(feature_columns2),
            num_layers=4
        )
        model_path15 = f"LSTM_{tag_prefix}2_{value}"
        trainer2 = FinancialTransformerTrainer(
            model=lstm_model2, model_path=model_path15, device=device
        )
        optimizer2 = optim.Adam(lstm_model2.parameters(), lr=0.001)
        trainer2.train_multiple_epochs(train_loader2, optimizer2, num_epochs=100, save_interval=100)

        models[f"train_dataset2{value}"] = train_dataset2
        models[f"trainer2{value}"] = trainer2
        models[f"model_path15_{value}"] = model_path15

        # --- Model 3: predict change_int using columnC features ---
        feature_columns3 = columnC
        target_column3 = "change_int"

        train_dataset3 = FinancialSequenceDataset(
            model, target_column3, feature_columns3, seq_len=12,
            scale_features=True, scale_target=False
        )
        scaler_tag3 = f"{tag_prefix}3"
        np.savetxt(f"120960/scaler_{scaler_tag3}_mean.csv",
                   train_dataset3.feature_scaler.mean_, delimiter=",")
        np.savetxt(f"120960/scaler_{scaler_tag3}_scale.csv",
                   train_dataset3.feature_scaler.scale_, delimiter=",")
        train_loader3 = DataLoader(train_dataset3, batch_size=32, shuffle=False)

        lstm_model3 = TimeSeriesLSTM(
            input_size=len(feature_columns3),
            hidden_size=len(feature_columns3),
            num_layers=4
        )
        model_path_colC = f"LSTM_{tag_prefix}3_{value}"
        trainer3 = FinancialTransformerTrainer(
            model=lstm_model3, model_path=model_path_colC, device=device
        )
        optimizer3 = optim.Adam(lstm_model3.parameters(), lr=0.001)
        trainer3.train_multiple_epochs(train_loader3, optimizer3, num_epochs=100, save_interval=100)

        models[f"train_dataset3{value}"] = train_dataset3
        models[f"trainer3{value}"] = trainer3
        models[f"model_path_colC_{value}"] = model_path_colC

    return models


# ──────────────────── Inference Helpers ────────────────────

def prepare_inference_df(total_df, target, buy_sell_label):
    """Prepare inference DataFrame matching the format expected by predict()."""
    df = total_df[total_df['timestamp'] % 5 == 0].copy()
    df.index = np.arange(len(df))

    shifts = int(target)
    df.insert(4, "change", df[f"change{target}"].shift(-shifts) * 100)
    df.insert(5, "Y_change15", df[f"change{12 * 15}"].shift(-int(shifts/4)) * 100)
    df.insert(6, "buy_sell", buy_sell_label)
    df.insert(7, "holding_time", 3600 * 1)
    df = df.drop(['high', 'low', 'open'], axis=1, errors='ignore')
    df.insert(7, "open_timestamp", df.timestamp)
    df.insert(8, "close_timestamp", df.timestamp.shift(-int(shifts)))
    df.insert(9, "open_date", df.snapshotTime)
    df.insert(10, "close_date", df.snapshotTime.shift(-int(shifts)))
    df["holding_time"] = df["close_timestamp"] - df["open_timestamp"]

    return df.dropna()


# ──────────────────── Main Loop ────────────────────

async def update_loop():
    pool = await init_db()

    # ── Load model_select ──
    try:
        model_select = np.load(asset_path('comp1_comp8_buy_seq60_5s.npy'))
    except FileNotFoundError:
        model_select = None
        print("[model_select.npy not found — continuing without]")

    # ── Period definitions ──
    total_periods_b = int(3600 * 2 / 5)
    columnB = [f'change{p}' for p in generate_periods(total_periods_b, target_step=12 * 60, long_step=60)]
    total_periods_c = int(3600 * 24 / 5)
    columnC = [f'change{p}' for p in generate_periods(total_periods_c, target_step=12 * 60, long_step=12 * 5)]
    model_length = [720, 1440]

    # ── Load initial raw data ──
    query_long = f"""
        SELECT "snapshotTime",
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {RAW_TABLE} ORDER BY "timestamp" DESC LIMIT {10000+total_period}
        ) AS sub ORDER BY "timestamp" ASC;
    """

    df4_old = await load_initial_data(pool, query_long)
    if not df4_old.empty:
        print("Old diff counts:")
        print(df4_old['timestamp'].diff(1)[1:].value_counts())

    if df4_old.empty:
        print("No data in btcusd_5s — exiting.")
        await pool.close()
        return

    # ── Load training data ──
    train_buy = await reload_train_data(pool, BUY_TABLE)
    train_sell = await reload_train_data(pool, SELL_TABLE)
    for name, tdf in [("buy", train_buy), ("sell", train_sell)]:
        if not tdf.empty:
            print(f"{name}_last_open: {tdf['open_date'].iloc[-1]}")

    # ── Model state ──
    pre_train = True
    buy_models = {}
    sell_models = {}
    last_open_time = None

    # ── Query for new data each loop ──
    query_new = f"""
        SELECT "snapshotTime",
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {RAW_TABLE}
            WHERE "timestamp" > $1::timestamptz
            ORDER BY "timestamp" DESC
        ) AS sub ORDER BY "timestamp" ASC;
    """

    # ── Main loop ──
    try:
        while True:
            start = time.time()

            # Query new rows since last known timestamp
            last_known_ts = datetime.fromtimestamp(int(df4_old['timestamp'].iloc[-1]), tz=timezone.utc)
            df_new = await load_initial_data(pool, query_new.replace("$1", f"'{last_known_ts.isoformat()}'"))

            # Update in-memory state
            if not df_new.empty:
                df4_old = pd.concat([df4_old, df_new], ignore_index=True)
                df4_old = df4_old.drop_duplicates(subset=['timestamp'])
                df4_old = df4_old.tail(total_period + 10000).copy()
                df4_old.index = np.arange(len(df4_old))
                last_row = df_new.iloc[-1]
                print(f"new 5s rows: {len(df_new)}, latest: {last_row['snapshotTime']} C={last_row['close']}")

            if df4_old.empty:
                end = time.time()
                cost = round(end - start, 1)
                sleep_sec = 5.5 - (start % 5)
                print(f"cost: {cost}")
                if cost < sleep_sec:
                    await asyncio.sleep(sleep_sec - cost)
                continue

            # ── Feature engineering ──
            total_df, periods, target_val = compute_features(df4_old)
            shifts_count = target_val
            target = 12 * 60
            shifts_count = target

            # ── Insert buy training data ──
            buy_last_open = train_buy['open_timestamp'].iloc[-1] if not train_buy.empty else 0
            buy_df = prepare_train_data(total_df, target, shifts_count, "buy")
            buy_new = buy_df[buy_df['open_timestamp'] > buy_last_open].copy()
            if not buy_new.empty:
                buy_new.index = np.arange(len(buy_new))
                await insert_train_batch(pool, buy_new, BUY_TABLE)
                print(f"inserted buy: {len(buy_new)} rows")
                train_buy = await reload_train_data(pool, BUY_TABLE)
                if not train_buy.empty:
                    print(f"buy_last_open: {train_buy['open_date'].iloc[-1]}")

            # ── Insert sell training data ──
            sell_last_open = train_sell['open_timestamp'].iloc[-1] if not train_sell.empty else 0
            sell_df = prepare_train_data(total_df, target, shifts_count, "sell")
            sell_df['change'] = -sell_df['change']
            sell_new = sell_df[sell_df['open_timestamp'] > sell_last_open].copy()
            if not sell_new.empty:
                sell_new.index = np.arange(len(sell_new))
                await insert_train_batch(pool, sell_new, SELL_TABLE)
                print(f"inserted sell: {len(sell_new)} rows")
                train_sell = await reload_train_data(pool, SELL_TABLE)
                if not train_sell.empty:
                    print(f"sell_last_open: {train_sell['open_date'].iloc[-1]}")

            # ── Pre-train / Retrain ──
            if pre_train:
                pre_train = False
                print("=== Pre-training BUY models ===")
                buy_train = await load_train_data_full(pool, BUY_TABLE)
                if not buy_train.empty:
                    buy_models = train_models(buy_train, model_length, is_buy=True, columnB=columnB, columnC=columnC)
                print("=== Pre-training SELL models ===")
                sell_train = await load_train_data_full(pool, SELL_TABLE)
                if not sell_train.empty:
                    sell_models = train_models(sell_train, model_length, is_buy=False, columnB=columnB, columnC=columnC)

                # Update last_open_time from training data
                if not buy_train.empty:
                    last_open_time = buy_train['open_timestamp'].values[-1]
                elif not sell_train.empty:
                    last_open_time = sell_train['open_timestamp'].values[-1]

            # ── Build inference DataFrame for spread check ──
            buy_inf = prepare_inference_df(total_df, target, "buy")
            sell_inf = prepare_inference_df(total_df, target, "sell")
            sell_inf['change'] = -sell_inf['change']

            last_timestamp = int(df4_old['timestamp'].values[-1])
            model_and_df8 = (last_timestamp - last_open_time) if last_open_time is not None else 0

            close_price = round(df4_old['close'].values[-1], 0)
            print(f"current time: {df4_old['snapshotTime'].values[-1]}, close: {close_price}, spread: {model_and_df8}")

            # ── Retrain trigger ──
            if model_and_df8 >= (3600 * 3):
                print("pre_train is set True (spread >= 3h)")
                pre_train = True

            # ── Inference ──
            elif model_and_df8 >= 3600:
                symbol = 'BTCUSD'

                # --- Buy inference ---
                if stop_market() and buy_models:
                    buy_vaild = buy_inf.tail(2).copy()
                    buy_vaild.loc[:, 'change'] = 0

                    for value in model_length:
                        trainer_key = f"trainer{value}"
                        trainer2_key = f"trainer2{value}"
                        trainer3_key = f"trainer3{value}"
                        ds_key = f"train_dataset{value}"
                        ds2_key = f"train_dataset2{value}"
                        ds3_key = f"train_dataset3{value}"

                        if trainer_key not in buy_models:
                            continue

                        check_buy = buy_models[trainer_key].predict(
                            test_df=buy_vaild, feature_columns_start=12, target_column='change',
                            feature_scaler=buy_models[ds_key].feature_scaler
                        )
                        check_buy2 = buy_models[trainer2_key].predict(
                            test_df=buy_vaild, feature_columns=columnB, target_column='change180',
                            feature_scaler=buy_models[ds2_key].feature_scaler
                        )
                        check_buy3 = buy_models[trainer3_key].predict(
                            test_df=buy_vaild, feature_columns=columnC, target_column='change180',
                            feature_scaler=buy_models[ds3_key].feature_scaler
                        )
                        print(f"buy model={value}: check_buy={check_buy}, check_buy2={check_buy2}, check_buy3={check_buy3}, "
                              f"time={buy_vaild.snapshotTime.values[-1]}, close={buy_vaild.close.values[-1]}")

                        if check_buy >= 0.70 and check_buy2 >= 0.70:
                            await insert_signal_async(pool, 'buy', symbol,
                                buy_vaild.close.values[-1], value, check_buy, check_buy2,
                                buy_vaild.open_timestamp.values[-1], singal3_val=check_buy3,
                                model_select_val=model_select)
                            print(f"Buy signal inserted: {check_buy}, {check_buy2}, {check_buy3}")
                        else:
                            await insert_signal_async(pool, 'buy', symbol,
                                buy_vaild.close.values[-1], value, 0, 0,
                                buy_vaild.open_timestamp.values[-1], singal3_val=0,
                                model_select_val=model_select)
                            print(f"No buy signal -- zeros inserted: {check_buy}, {check_buy2}")

                # --- Sell inference ---
                if stop_market() and sell_models:
                    sell_vaild = sell_inf.tail(2).copy()
                    sell_vaild.loc[:, 'change'] = 0

                    last_val = model_length[-1]
                    trainer_key = f"trainer{last_val}"
                    trainer2_key = f"trainer2{last_val}"
                    trainer3_key = f"trainer3{last_val}"
                    ds_key = f"train_dataset{last_val}"
                    ds2_key = f"train_dataset2{last_val}"
                    ds3_key = f"train_dataset3{last_val}"

                    if trainer_key in sell_models:
                        check_sell = sell_models[trainer_key].predict(
                            test_df=sell_vaild, feature_columns_start=12, target_column='change',
                            feature_scaler=sell_models[ds_key].feature_scaler
                        )
                        check_sell2 = sell_models[trainer2_key].predict(
                            test_df=sell_vaild, feature_columns=columnB, target_column='Y_change15',
                            feature_scaler=sell_models[ds2_key].feature_scaler
                        )
                        check_sell3 = sell_models[trainer3_key].predict(
                            test_df=sell_vaild, feature_columns=columnC, target_column='Y_change15',
                            feature_scaler=sell_models[ds3_key].feature_scaler
                        )
                        print(f"sell: check={check_sell}, check2={check_sell2}, check3={check_sell3}, "
                              f"close={close_price}, time={sell_vaild.open_date.values[-1]}")

                        if check_sell >= 0.21 and check_sell2 >= 0.41:
                            await insert_signal_async(pool, 'sell', symbol,
                                sell_vaild.close.values[-1], last_val, check_sell, check_sell2,
                                sell_vaild.open_timestamp.values[-1], singal3_val=check_sell3,
                                model_select_val=model_select)
                            print(f"Sell signal inserted: {check_sell}, {check_sell2}, {check_sell3}")
                        else:
                            await insert_signal_async(pool, 'sell', symbol,
                                sell_vaild.close.values[-1], last_val, 0, 0,
                                sell_vaild.open_timestamp.values[-1], singal3_val=0,
                                model_select_val=model_select)
                            print(f"No sell signal -- zeros inserted: {check_sell}, {check_sell2}")

            # ── Sleep until next 5s boundary ──
            end = time.time()
            sleep_sec = 5.5 - (start % 5)
            cost = round(end - start, 1)
            print(f"cost: {cost}")
            if cost < sleep_sec:
                await asyncio.sleep(sleep_sec - cost)
    finally:
        await pool.close()


# ──────────────────── Entry Point ────────────────────

if __name__ == "__main__":
    asyncio.run(update_loop())
