import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import datetime as raw_datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

import psycopg2
import psycopg2.extras
from collections import namedtuple

from db_env import asset_path
from feature_method import bin_change15_int_series, bin_change_int_series

# ─── Configuration ───
from db_env import db_config

DB_CONFIG = db_config()

table_symbol = "btcusd"
BUY_TABLE = f"{table_symbol}_17280_BUY720_336_5s"
SELL_TABLE = f"{table_symbol}_17280_SELL720_336_5s"
BUY_SIGNAL_TABLE = f"{table_symbol}_17280_BUY720_336_5s_singal"
SELL_SIGNAL_TABLE = f"{table_symbol}_17280_SELL720_336_5s_singal"

TrainingResult = namedtuple('TrainingResult', ['model', 'train_losses'])
SequenceData = namedtuple('SequenceData', ['features', 'target'])

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
        if 'snapshottime' in df.columns:
            df = df.rename(columns={'snapshottime': 'snapshotTime'})
        if 'snapshotTime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['snapshotTime']):
            if df['snapshotTime'].dt.tz is not None:
                df['snapshotTime'] = df['snapshotTime'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"Database error: {e}")
        raise
    finally:
        if connection:
            connection.close()


def create_signal_tables(db_config=None):
    if db_config is None:
        db_config = DB_CONFIG
    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        for table_name in [BUY_SIGNAL_TABLE, SELL_SIGNAL_TABLE]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT,
                    close FLOAT8,
                    check_buy FLOAT8,
                    check_buy2 FLOAT8,
                    check_buy3 FLOAT8 DEFAULT 0,
                    open_timestamp BIGINT,
                    model_select TEXT DEFAULT NULL
                );
            """)
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS check_buy3 FLOAT8 DEFAULT 0;
                """)
            except Exception:
                pass
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS close FLOAT8;
                """)
            except Exception:
                pass
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS model_select TEXT DEFAULT NULL;
                """)
            except Exception:
                pass
        connection.commit()
    except Exception as e:
        print(f"Error creating signal tables: {e}")
    finally:
        if connection:
            connection.close()


def trim_signal_table(db_config, table_name, max_rows=3600):
    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute(f"""
            DELETE FROM {table_name}
            WHERE open_timestamp < (
                SELECT MIN(open_timestamp) FROM (
                    SELECT open_timestamp FROM {table_name}
                    ORDER BY open_timestamp DESC
                    LIMIT {max_rows}
                ) AS latest
            );
        """)
        connection.commit()
    except Exception as e:
        print(f"Error trimming signal table: {e}")
    finally:
        if connection:
            connection.close()


def insert_signal(db_config, table_name, symbol_val, close_val, check_buy_val, check_buy2_val, open_ts, check_buy3_val=0, model_select_val=None):
    connection = None
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        model_str = str(model_select_val) if model_select_val is not None else None
        cursor.execute(f"""
            INSERT INTO {table_name} (symbol, close, check_buy, check_buy2, check_buy3, open_timestamp, model_select)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (symbol_val, float(close_val), float(check_buy_val), float(check_buy2_val), float(check_buy3_val), int(open_ts), model_str))
        connection.commit()
    except Exception as e:
        print(f"Error inserting signal: {e}")
    finally:
        if connection:
            connection.close()
    trim_signal_table(db_config, table_name)


def stop_market():
    return True

    year, month, day, hour, minute = time.localtime(time.time())[:5]
    current_date = raw_datetime.date(year, month, day)
    if current_date.weekday() >= 5 and hour == 4:
        return False
    else:
        return True


# ─── ML Classes ───

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

    def train_multiple_epochs(self, train_loader, optimizer, num_epochs=10, save_dir='checkpoints', save_interval=50):
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

    def load_model_by_epoch(self, epoch, save_dir='checkpoints'):
        save_path = os.path.join(save_dir, f'{self.model_path}{epoch}.pth')
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint not found at {save_path}")
        checkpoint = torch.load(save_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Loaded model from epoch {checkpoint['epoch']} with loss: {checkpoint['loss']:.4f}")
        return checkpoint

    def predict(self, test_df, epoch=None, save_dir='checkpoints',
                feature_columns_start=11, target_column='change', seq_len=1,
                feature_scaler=None):
        if epoch is not None:
            self.load_model_by_epoch(epoch, save_dir)
        self.model.eval()
        self.model.to(self.device)
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
        avg_loss = total_loss / len(test_dataset)
        y_pred = []
        for sublist in all_predictions:
            if isinstance(sublist, (list, np.ndarray)):
                for element in sublist:
                    y_pred.append(float(element))
            else:
                y_pred.append(float(sublist))
        y_pred_value = round(y_pred[-1], 4)
        return y_pred_value

    def predict_single(self, sample_data, epoch=None, save_dir='checkpoints'):
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


# ─── Main Loop ───

global variable_rate, df4_temp, record_list1, train_old, train, df4_old, df4_train, model, df4, column

last_create_time = 0
variable_rate = 1

record_list = []
call_list = []
put_list = []
call_stop_loss_list = []
put_stop_loss_list = []
time_list = []
moving_list = []
stop_condition = []
volume_list = {}

money = 10000
trade = 0
commission_fee = 0.001
leverage = 100 * 1
price_spread = 1

# ── Load initial OHLCV data ──
while True:
    try:
        query = f"""
        SELECT "snapshotTime",
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {table_symbol}_5s
            ORDER BY "timestamp" DESC
            LIMIT 25000
        ) AS sub
        ORDER BY timestamp ASC;
        """
        df4_old = fetch_data_from_database(query)
        print(df4_old['timestamp'].diff(1)[1:].value_counts())
        break
    except Exception as e:
        print(e)
        time.sleep(10)

train_record = 0
order_set = True
trading_test = 0

Data_1000_list = {}
timestamp_list = {}

is_buy = True
model_length = [720]
model_list = {}
levearge = 100

pre_train = True

try:
    model_select = np.load(asset_path('comp1_comp8_buy_seq60_5s.npy'))
except FileNotFoundError:
    model_select = None
    print("[model_select.npy not found — continuing without]")

if is_buy:
    for value in model_length:
        model_list[f"model_path60_{value}"] = f"LSTM_A1_{value}"
        model_list[f"model_path15_{value}"] = f"LSTM_A2_{value}"
    table_name = BUY_TABLE
else:
    for value in model_length:
        model_list[f"model_path60_{value}"] = f"LSTM_B1_{value}"
        model_list[f"model_path15_{value}"] = f"LSTM_B1_{value}"
    table_name = SELL_TABLE

# ── Periods ──
periods = [12]
current = 12
step = 12
total = int(3600 * 24 / 5)
target = 12 * 60

while current < total:
    current += step
    if current <= total:
        periods.append(current)
    if current >= target:
        step = 60

columnB = [f'change{period}' for period in periods]

# Ensure signal tables exist
create_signal_tables()

# ── Main prediction loop ──
while True:
    start_time = time.time()

    if pre_train:
        while True:
            try:
                query = f"""
                SELECT DISTINCT * FROM (
                    SELECT DISTINCT * FROM {table_name}
                    ORDER BY open_timestamp DESC
                    LIMIT 10000
                ) AS last_rows
                ORDER BY open_timestamp ASC;
                """
                train = fetch_data_from_database(query)
                for col in ['open_date', 'close_date']:
                    if col in train.columns and pd.api.types.is_datetime64_any_dtype(train[col]):
                        if train[col].dt.tz is not None:
                            train[col] = train[col].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
                break
            except Exception as e:
                print(e)
                time.sleep(10)

    # Fetch latest OHLCV data
    while True:
        try:
            query = f"""
            SELECT "snapshotTime",
                   extract(epoch from "timestamp")::bigint AS timestamp,
                   open, high, low, close, volume
            FROM (
                SELECT * FROM {table_symbol}_5s
                ORDER BY "timestamp" DESC
                LIMIT 1000
            ) AS sub
            ORDER BY timestamp ASC;
            """
            df4_temp = fetch_data_from_database(query)
            break
        except Exception as e:
            print(e)
            time.sleep(1)

    # ── Pre-train models ──
    if pre_train:
        pre_train = False

        train.insert(4, "change_int", 0, allow_duplicates=False)
        shifts = 12 * 15
        train.insert(5, f"Y_change15", train[f"change{shifts}"].shift(-shifts) * 100)
        train.insert(6, "change15_int", 0, allow_duplicates=False)

        while len(train) > 0 and train.iloc[-1]['open_timestamp'] % (3600 * 2) != 0:
            train = train.iloc[:-1]

        train = train.copy()[train['open_timestamp'] % 5 == 0]

        for value in model_length:
            model = train.tail(value).copy()
            model.index = np.arange(len(model))
            model_list[f"model{value}"] = model

        for value in model_length:
            model = model_list[f"model{value}"].copy()
            model_path60 = model_list[f"model_path60_{value}"]
            model_path15 = model_list[f"model_path15_{value}"]

            # Discrete binning moved to ../feature_method.py (private).
            model['change_int'] = bin_change_int_series(model['change'])

            x_train = model.iloc[:, 14:]
            y_train = model["change_int"]

            print("shape:", model.shape, end=" ,")
            print(model.columns[:15].tolist())
            feature_columns = model.columns[14:].tolist()
            target_column = "change_int"

            train_dataset = FinancialSequenceDataset(
                model, target_column, feature_columns, seq_len=12,
                scale_features=True, scale_target=False
            )
            # Save scaler to CSV for Rust inference (use env.txt naming: scaler_A1)
            os.makedirs("checkpoints", exist_ok=True)
            scaler_tag = "A1" if is_buy else "B1"
            np.savetxt(f"checkpoints/scaler_{scaler_tag}_mean.csv",
                       train_dataset.feature_scaler.mean_, delimiter=",")
            np.savetxt(f"checkpoints/scaler_{scaler_tag}_scale.csv",
                       train_dataset.feature_scaler.scale_, delimiter=",")
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

            LSTMmodel = TimeSeriesLSTM(
                input_size=len(feature_columns),
                hidden_size=len(feature_columns),
                num_layers=4
            )
            trainer = FinancialTransformerTrainer(
                model=LSTMmodel,
                model_path=model_path60,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            optimizer = optim.Adam(LSTMmodel.parameters(), lr=0.001)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            result = trainer.train_multiple_epochs(train_loader, optimizer, num_epochs=100, save_interval=100)

            model_list[f"train_dataset{value}"] = train_dataset
            model_list[f"train_loader{value}"] = train_loader
            model_list[f"LSTMmodel{value}"] = LSTMmodel
            model_list[f"trainer{value}"] = trainer
            model_list[f"optimizer{value}"] = optimizer
            model_list[f"result{value}"] = result

            # ── Second model ──
            # Discrete binning moved to ../feature_method.py (private).
            model['change15_int'] = bin_change15_int_series(model['Y_change15'])

            feature_columns2 = columnB
            target_column2 = "change15_int"

            train_dataset2 = FinancialSequenceDataset(
                model, target_column2, feature_columns2, seq_len=12,
                scale_features=True, scale_target=False
            )
            # Save model 2 scaler to CSV for Rust inference
            scaler_tag = "A2" if is_buy else "B2"
            np.savetxt(f"checkpoints/scaler_{scaler_tag}_mean.csv",
                       train_dataset2.feature_scaler.mean_, delimiter=",")
            np.savetxt(f"checkpoints/scaler_{scaler_tag}_scale.csv",
                       train_dataset2.feature_scaler.scale_, delimiter=",")
            train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=False)

            LSTMmodel2 = TimeSeriesLSTM(
                input_size=len(feature_columns2),
                hidden_size=len(feature_columns2),
                num_layers=4
            )
            trainer2 = FinancialTransformerTrainer(
                model=LSTMmodel2,
                model_path=model_path15,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            optimizer2 = optim.Adam(LSTMmodel2.parameters(), lr=0.001)
            result2 = trainer2.train_multiple_epochs(train_loader2, optimizer2, num_epochs=100, save_interval=100)

            model_list[f"train_dataset2{value}"] = train_dataset2
            model_list[f"train_loader2{value}"] = train_loader2
            model_list[f"LSTMmodel2{value}"] = LSTMmodel2
            model_list[f"trainer2{value}"] = trainer2
            model_list[f"optimizer2{value}"] = optimizer2
            model_list[f"result2{value}"] = result2

    # ── Update OHLCV state ──
    lastest_model = model['open_date'].values[-1]

    df4_old = pd.concat([df4_old.copy(), df4_temp.copy()], ignore_index=True)
    df4_old = df4_old.drop_duplicates(subset=['timestamp'])
    df4_old.index = np.arange(len(df4_old))

    lastest_model_open = int(model['open_timestamp'].values[-1])
    lastest_model_close = int(model['close_timestamp'].values[-1])

    last_timestamp = int(df4_old['timestamp'].values[-1])
    last_open_time = model['open_timestamp'].values[-1]
    model_and_df8 = (last_timestamp - last_open_time)

    df_1min = df4_old.copy()

    close_price_1min = round(df_1min.close.values[-1], 0)
    last_price_1min = df_1min.close.values[-2]
    low_price_1min = df_1min.low.values[-2]
    max_price_1min = df_1min.high.values[-2]
    mid_price_1min = (low_price_1min + max_price_1min) / 2
    time_1min = df_1min.timestamp.values[-1]

    print('latest model:', lastest_model, 'current time:', df4_old['snapshotTime'].values[-1], close_price_1min, end=" ")

    # ── Feature engineering ──
    total_df = df_1min
    close_series = total_df['close']

    periods = [12]
    current = 12
    step = 12
    total = int(3600 * 24 / 5) * 1
    target = 12 * 60

    while current < total:
        current += step
        if current <= total:
            periods.append(current)
        if current >= target:
            step = 60

    shifts = pd.DataFrame({
        f'shift_{period}': close_series.shift(period)
        for period in periods
    })
    changes = np.round((close_series.values[:, None] - shifts.values) / shifts.values * 100, 4)
    changes_df = pd.DataFrame(
        changes,
        columns=[f'change{period}' for period in periods],
        index=total_df.index
    )
    total_df = pd.concat([total_df, changes_df], axis=1)
    total_df = total_df.dropna()

    total_df1 = total_df[total_df['timestamp'] % 5 == 0]
    total_df1.index = np.arange(len(total_df1))

    shifts = int(target / 1)
    total_df1.insert(4, "change", total_df1[f"change{target}"].shift(-shifts) * 100)
    total_df1.insert(5, "buy_sell", "buy")
    total_df1.insert(6, "holding_time", 3600 * 1)
    total_df1 = total_df1.drop(['high', 'low', 'open', 'open'], axis=1)
    total_df1.insert(6, "open_timestamp", total_df1.timestamp)
    total_df1.insert(7, "close_timestamp", total_df1.timestamp.shift(-int(shifts)))
    total_df1.insert(8, "open_date", total_df1.snapshotTime)
    total_df1.insert(9, "close_date", total_df1.snapshotTime.shift(-int(shifts)))
    total_df1["holding_time"] = total_df1["close_timestamp"] - total_df1["open_timestamp"]
    total_df1.insert(5, "Y_change15", total_df1[f"change{12 * 15}"].shift(-shifts) * 100)

    print('model spread:', model_and_df8)

    if not is_buy:
        total_df1['change'] = -total_df1['change']
        total_df1['buy_sell'] = "sell"

    # ── Check retrain condition ──
    if model_and_df8 >= (3600 * 3):
        print('pre_train is set True')
        pre_train = True
        check_buy = 0

    elif model_and_df8 >= 3600:
        vaild_df = total_df1.tail(2).copy()
        vaild_df.loc[:, 'change'] = 0

        symbol = 'BTCUSD'

        if stop_market() and is_buy:
            for value in model_length:
                check_buy = model_list[f"trainer{value}"].predict(
                    test_df=vaild_df, feature_columns_start=12, target_column='change',
                    feature_scaler=model_list[f"train_dataset{value}"].feature_scaler
                )
                check_buy2 = model_list[f"trainer2{value}"].predict(
                    test_df=vaild_df[columnB], feature_columns_start=0, target_column='change180',
                    feature_scaler=model_list[f"train_dataset2{value}"].feature_scaler
                )
                print("model:", value, check_buy, check_buy2,
                      vaild_df.snapshotTime.values[-1], vaild_df.close.values[-1])

                if check_buy >= 0.70 and check_buy2 >= 0.70:
                    insert_signal(DB_CONFIG, BUY_SIGNAL_TABLE, symbol,
                                  vaild_df.close.values[-1],
                                  check_buy, check_buy2, vaild_df.open_timestamp.values[-1], check_buy3_val=0,
                                  model_select_val=model_select)
                    print(f"Buy signal inserted: {check_buy}, {check_buy2}")
            else:
                    insert_signal(DB_CONFIG, BUY_SIGNAL_TABLE, symbol,
                                  vaild_df.close.values[-1],
                                  check_buy, check_buy2, vaild_df.open_timestamp.values[-1], check_buy3_val=0,
                                  model_select_val=model_select)
                    print(f"No buy signal — original values inserted: {check_buy}, {check_buy2}")

        elif stop_market() and not is_buy:
            last_val = model_length[-1]
            check_sell = model_list[f"trainer{last_val}"].predict(
                test_df=vaild_df, feature_columns_start=12, target_column='change',
                feature_scaler=model_list[f"train_dataset{last_val}"].feature_scaler
            )
            check_sell2 = model_list[f"trainer2{last_val}"].predict(
                test_df=vaild_df[columnB], feature_columns_start=0, target_column='change15',
                feature_scaler=model_list[f"train_dataset2{last_val}"].feature_scaler
            )
            print(f"sell:", check_sell, "sell2:", check_sell2,
                  ",720-2160:", vaild_df.shape, close_price_1min, end=" ")
            print("last vaild_df:", vaild_df.open_date.values[-1])

            if check_sell >= 0.21 and check_sell2 >= 0.41:
                insert_signal(DB_CONFIG, SELL_SIGNAL_TABLE, symbol,
                              vaild_df.close.values[-1],
                              check_sell, check_sell2, vaild_df.open_timestamp.values[-1], check_buy3_val=0,
                              model_select_val=model_select)
                print(f"Sell signal inserted: {check_sell}, {check_sell2}")
            else:
                insert_signal(DB_CONFIG, SELL_SIGNAL_TABLE, symbol,
                              vaild_df.close.values[-1],
                              check_sell, check_sell2, vaild_df.open_timestamp.values[-1], check_buy3_val=0,
                              model_select_val=model_select)
                print(f"No sell signal — original values inserted: {check_sell}, {check_sell2}")

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
