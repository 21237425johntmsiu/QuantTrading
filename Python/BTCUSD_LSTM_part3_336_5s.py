import pandas as pd
import numpy as np

import os

import time
from datetime import datetime
import datetime as raw_datetime
import threading

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler


import psycopg2
import psycopg2.extras

from collections import namedtuple
from db_env import asset_path, db_config
from feature_method import bin_change15_int_series, bin_change_int_series

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    print("[MT5 not available — trading functions disabled]")

# model_path = "LSTM_B_"

if True:
        
        def buy_order(symbol,volume):
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
                    print("Failed to get XAUUSD price")
                    return None
        
                current_price = tick.ask # Using bid price for sell orders
                # stop_loss = current_price * stop  # 1% above entry (for sell orders)
                # take_profit = current_price * profit  # 10% below entry
        
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": current_price,
                    # "sl": stop_loss,
                    # "tp": take_profit,
                    "deviation": 20,  # Wider deviation for commodities
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
        
        def sell_order(symbol,volume):
            global mt5
            if mt5 is None:
                print("MT5 not available, skipping sell_order")
                return None
            try:
                if not mt5.symbol_select(symbol, True):
                    print(f"Add {symbol} to Market Watch first!")
                    return None
                    
                tick = mt5.symbol_info_tick(symbol)
                if not tick or not tick.bid:  # Using bid price for sell orders
                    print("Failed to get XAUUSD price")
                    return None
        
                current_price = tick.bid  # Using bid price for sell orders
        
        
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_SELL,  # Changed to SELL
                    "price": current_price,
            
                    "deviation": 20,  # Wider deviation for commodities
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
            # Initialize MT5 connection
            if not mt5.initialize():
                print("MT5 initialization failed, error code =", mt5.last_error())
                return False
            
            # MT5 credentials are configured in env.txt (repo root)
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
            """Close an open position by its ticket number"""
            try:
                # Get the position by ticket number
                position = mt5.positions_get(ticket=ticket)
                
                if not position:
                    print(f"No position found with ticket #{ticket}")
                    return False
                
                position = position[0]  # Get the position object
                
                # Determine the closing price
                symbol_info = mt5.symbol_info(position.symbol)
                if not symbol_info:
                    print(f"Failed to get market data for {position.symbol}")
                    return False
                    
                if position.type == mt5.ORDER_TYPE_BUY:
                    price = symbol_info.bid  # Use bid price to close long positions
                    order_type = mt5.ORDER_TYPE_SELL
                else:
                    price = symbol_info.ask  # Use ask price to close short positions
                    order_type = mt5.ORDER_TYPE_BUY
                
                # Prepare the close request
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
                
                # Send the close order
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Successfully closed position #{ticket}",end = " ")
                    print(f"Symbol: {position.symbol}",end = " ")
                    print(f"Type: {'BUY' if position.type == 0 else 'SELL'}",end = " ")
                    print(f"Volume: {position.volume}",end = " ")
                    print(f"Close Price: {price}",end = " ")
                    print(f"Profit: {position.profit}",end = " ")
                    # print(f"Close Time: {datetime.now()}")
                    print(f"Close Time: {datetime.fromtimestamp(position.time - 3600*7)}")
               
                    return True
                else:
                    print(f"Failed to close position #{ticket}")
                    print(f"Error code: {result.retcode}")
                    print(f"Error description: {mt5.last_error()}")
                    return False
        
                print(result)
            
            except Exception as e:
                print(f"Error closing position: {str(e)}")
                return False
                
        class TimeSeriesLSTM(nn.Module):
            
            def __init__(self, input_size, hidden_size=64, num_layers=3, 
                         output_size=1):
                """
                LSTM model for financial forecasting
                """
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # LSTM layer
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True
                )
                
                # Output layers
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                # LSTM layer
                lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]
                
                # Use the last time step for prediction
                last_hidden = lstm_out[:, -1, :]
                return self.fc(last_hidden)
        
            
        class FinancialSequenceDataset(Dataset):
            
            def __init__(self, dataframe, target_column, feature_columns, seq_len=60,
                         scale_features=True, scale_target=False,
                         feature_scaler=None, target_scaler=None):
                """
                Dataset for financial time series with optional scaling.
                If external scalers are provided, they are used for transformation;
                otherwise new scalers are fitted on the data.
                """
                self.dataframe = dataframe.copy()
                self.target_column = target_column
                self.feature_columns = feature_columns
                self.seq_len = seq_len
        
                # Extract raw values
                features_raw = self.dataframe[self.feature_columns].values
                target_raw = self.dataframe[self.target_column].values
        
                # --- Feature scaling ---
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
        
                # --- Target scaling ---
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
                x_seq = self.features[idx:idx+self.seq_len]
                y_target = self.target[idx+self.seq_len-1]
        
                return SequenceData(
                    features=torch.FloatTensor(x_seq),
                    target=torch.FloatTensor([y_target])
                )
        
        
        
        class FinancialTransformerTrainer:
            
            def __init__(self, model, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
                """
                Initialize trainer with model and model path
                
                Args:
                    model: The PyTorch model
                    model_path: Base path/prefix for saving model checkpoints (e.g., 'model_epoch')
                    device: Device to run model on
                """
                self.model = model.to(device)
                self.device = device
                self.criterion = nn.MSELoss()
                self.model_path = model_path
            
            def train_multiple_epochs(self, train_loader, optimizer, num_epochs=10, save_dir='checkpoints', save_interval=50):
                """
                Train model and save checkpoints at regular intervals
                
                Args:
                    train_loader: DataLoader for training data
                    optimizer: PyTorch optimizer
                    num_epochs: Number of epochs to train
                    save_dir: Directory to save checkpoints
                    save_interval: Save checkpoint every N epochs
                """
                self.model.train()
                train_losses = []
                
                # Create save directory if it doesn't exist
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
                    
                    # Save model at specified intervals
                    if (epoch + 1) % save_interval == 0:
                        save_path = os.path.join(save_dir, f'{self.model_path}_{epoch + 1}.pth')
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                        }, save_path)
                        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f} - Model saved to {save_path}")
                    # else:
                    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
                
                return TrainingResult(self.model, train_losses)
            
            def load_model_by_epoch(self, epoch, save_dir='checkpoints'):
                """
                Load a saved model checkpoint by epoch number
                
                Args:
                    epoch: Epoch number to load
                    save_dir: Directory where checkpoints are saved
                """
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
                
                """
                Evaluate model on test data and return prediction using self.model_path
                
                Args:
                    test_df: DataFrame with test data
                    epoch: Optional epoch number to load specific model checkpoint (uses current model if None)
                    save_dir: Directory where checkpoints are saved
                    feature_columns_start: Index where feature columns start (default: 11)
                    target_column: Name of target column (default: 'change')
                    seq_len: Sequence length for test dataset (default: 1)
                    feature_scaler: Fitted scaler from training data (e.g., train_dataset.feature_scaler)
                
                Returns:
                    float: Predicted value for the last sample
                """
                # Load specific model checkpoint if epoch is provided
                if epoch is not None:
                    self.load_model_by_epoch(epoch, save_dir)
                
                self.model.eval()
                self.model.to(self.device)
                
                # Prepare test dataset
                feature_columns = test_df.columns[feature_columns_start:].tolist()
                
                test_dataset = FinancialSequenceDataset(
                    test_df, 
                    target_column, 
                    feature_columns, 
                    seq_len=seq_len, 
                    scale_features=True,
                    feature_scaler=feature_scaler      # 使用训练时的scaler
                )
                
                all_predictions = []
                all_targets = []
                total_loss = 0.0
                
                with torch.no_grad():
                    for sample in test_dataset:
                        # 正确访问 SequenceData 的属性
                        batch_x = sample.features.to(self.device)
                        batch_y = sample.target.to(self.device)
                        
                        # Add batch dimension if sequence length is 1
                        if seq_len == 1:
                            batch_x = batch_x.unsqueeze(0)   # (1, seq_len, n_features)
                            batch_y = batch_y.unsqueeze(0)   # (1, 1)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        total_loss += loss.item() * batch_x.size(0)
                        
                        all_predictions.extend(outputs.cpu().numpy())
                        all_targets.extend(batch_y.cpu().numpy())
                
                # 使用 dataset 的实际长度计算平均损失
                avg_loss = total_loss / len(test_dataset)
                
                # Flatten predictions
                y_pred = []
                for sublist in all_predictions:
                    if isinstance(sublist, (list, np.ndarray)):
                        for element in sublist:
                            y_pred.append(float(element))
                    else:
                        y_pred.append(float(sublist))
                
                # Return the last prediction
                y_pred_value = round(y_pred[-1], 4)
                
                # 可选：打印详细信息（调试用）
                # print(f"Prediction completed - Test Loss: {avg_loss:.4f}, Last Prediction: {y_pred_value}")
                
                return y_pred_value
            
            def predict_single(self, sample_data, epoch=None, save_dir='checkpoints'):
                """
                Predict for a single sample (useful for real-time predictions)
                
                Args:
                    sample_data: Single sample with same feature columns as training
                    epoch: Optional epoch to load
                    save_dir: Directory where checkpoints are saved
                
                Returns:
                    float: Prediction value
                """
                if epoch is not None:
                    self.load_model_by_epoch(epoch, save_dir)
                
                self.model.eval()
                
                # Convert to tensor and add batch and sequence dimensions
                if isinstance(sample_data, np.ndarray):
                    sample_tensor = torch.FloatTensor(sample_data)
                elif isinstance(sample_data, pd.DataFrame):
                    sample_tensor = torch.FloatTensor(sample_data.values)
                else:
                    sample_tensor = torch.FloatTensor(sample_data)
                
                # Add batch dimension and sequence dimension if needed
                if sample_tensor.dim() == 1:
                    sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
                elif sample_tensor.dim() == 2:
                    sample_tensor = sample_tensor.unsqueeze(0)
                
                sample_tensor = sample_tensor.to(self.device)
                
                with torch.no_grad():
                    prediction = self.model(sample_tensor)
                
                return prediction.cpu().item()
            


        DB_CONFIG = db_config()

        def fetch_data_from_database(query, db_config=None, return_type='dataframe'):
            """
            Fetch data from TimescaleDB (PostgreSQL) with error handling.

            Parameters:
            - query: SQL query string to execute
            - db_config: Dictionary containing database connection parameters
            - return_type: 'dataframe' (default) or 'raw'

            Returns:
            - Depending on return_type: DataFrame or raw cursor
            """
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

                # PostgreSQL folds unquoted identifiers to lowercase, so "time AS snapshotTime"
                # becomes "snapshottime". Rename to match the camelCase used in the code.
                if 'snapshottime' in df.columns:
                    df = df.rename(columns={'snapshottime': 'snapshotTime'})

                # Convert timezone-aware snapshotTime to naive HKT (Asia/Shanghai)
                # so .values[-1] display doesn't silently convert to UTC.
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
            """Create signal tables if they don't exist."""
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
                            check_buy FLOAT8,
                            check_buy2 FLOAT8,
                            open_timestamp BIGINT
                        );
                    """)
                connection.commit()
            except Exception as e:
                print(f"Error creating signal tables: {e}")
            finally:
                if connection:
                    connection.close()

        def insert_signal(db_config, table_name, symbol_val, check_buy_val, check_buy2_val, open_ts):
            """Insert a signal record into the signal table."""
            connection = None
            try:
                connection = psycopg2.connect(**db_config)
                cursor = connection.cursor()
                cursor.execute(f"""
                    INSERT INTO {table_name} (symbol, check_buy, check_buy2, open_timestamp)
                    VALUES (%s, %s, %s, %s)
                """, (symbol_val, float(check_buy_val), float(check_buy2_val), int(open_ts)))
                connection.commit()
            except Exception as e:
                print(f"Error inserting signal: {e}")
            finally:
                if connection:
                    connection.close()

        def stop_market():

                return True
            
                year,month,day,hour,minute = time.localtime(time.time())[:5]
            
                current_date = raw_datetime.date(year, month, day)
                # print(current_date.weekday())
                # year,month,day,hour,minute = time.localtime(time.time())[:5]

                if current_date.weekday() >= 5 and hour == 4 :
                    return False

                # elif 3 <= hour < 8:    
                #     return False

                # elif hour == 8 and minute > 15 :
                #     return True

                # elif hour == 11 and minute >= 55:
                #     return False

                # elif hour == 12:
                #     return False

                # elif hour == 4 and minute >= 25:
                #     return False

                # elif hour == 5 and minute >= 30:
                #     return True

                else: return True
                    
        global variable_rate,df4_temp,record_list1,train_old,train,df4_old, df4_train,model,df4,column
        
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

        order_id = []
        order_df = pd.DataFrame()
        
        table_symbol = "btcusd"

        BUY_SIGNAL_TABLE = f"{table_symbol}_17280_BUY720_336_5s_singal"
        SELL_SIGNAL_TABLE = f"{table_symbol}_17280_SELL720_336_5s_singal"

        TrainingResult = namedtuple('TrainingResult', ['model', 'train_losses'])
        SequenceData = namedtuple('SequenceData', ['features', 'target'])

        while True:
            #with lock:
                try:
               
                        # query = f"""
                        # SELECT DISTINCT * FROM (
                        #     SELECT DISTINCT * FROM {table_symbol}_5s
                        #     ORDER BY timestamp DESC
                        #     LIMIT 1000
                        # ) AS last_rows
                        # ORDER BY timestamp ASC;
                        # """
                        # df4_temp = fetch_data_from_database(query)
                        # #print(df4_temp['timestamp'].diff(1)[1:].value_counts())
                        # print(datetime.fromtimestamp(round(time.time(),0)),end = " ")
                    
                        #time.sleep(round(time.time(),0)%20)
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

        order_list = {
            "order_ID":[],
            "start_time":[],
        }

        is_buy = True
        model_length = [720]
        model_list = {
         
               
        }
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
            # model_path1 = "LSTM_A1_"
            # model_path2 = "LSTM_A2_"
            table_name = f"{table_symbol}_17280_BUY720_336_5s"
            if connect_to_mt5():
            
            # buy_positions = [pos for pos in mt5.positions_get() if pos.type == mt5.ORDER_TYPE_SELL]
                buy_positions = [pos for pos in mt5.positions_get()]
                
                for buy_position in buy_positions:
                    if "BUY" in buy_position.comment:
                            
                        order_list["order_ID"].append(buy_position.ticket)
                        order_list["start_time"].append(buy_position.time + 3600*5)
                
        else:
            for value in model_length:
                model_list[f"model_path60_{value}"] = f"LSTM_B1_{value}"
                model_list[f"model_path15_{value}"] = f"LSTM_B1_{value}"
                
            # model_path1 = "LSTM_B1_"
            # model_path2 = "LSTM_B2_"
            
            table_name = f"{table_symbol}_17280_SELL720_336_5s"
            if connect_to_mt5():
            
            # buy_positions = [pos for pos in mt5.positions_get() if pos.type == mt5.ORDER_TYPE_SELL]
                buy_positions = [pos for pos in mt5.positions_get()]
            
                for buy_position in buy_positions:
                    if "SELL" in buy_position.comment:
                        
                        order_list["order_ID"].append(buy_position.ticket)
                        order_list["start_time"].append(buy_position.time + 3600*5)
            
        # connect_to_mt5() uses MT5_* credentials from env.txt (repo root)
        print(order_list)
                
        
        # for i in range(1):

        # model_length = [360,720,1440,2160]
        # model_list = {
         
               
        # }
        periods = [12]
        current = 12
        step = 12 # First step from 5 to 30
        
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

                            # Convert timezone-aware date columns to naive HKT
                            for col in ['open_date', 'close_date']:
                                if col in train.columns and pd.api.types.is_datetime64_any_dtype(train[col]):
                                    if train[col].dt.tz is not None:
                                        train[col] = train[col].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

                            # print(train['open_timestamp'].diff(1)[1:].value_counts())
        
                            #time.sleep(round(time.time(),0)%20)
                            break
        
        
                    except Exception as e:
                        print(e)
                        time.sleep(10)
               
            # start_time = time.time()

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
                        # print(df4_temp['timestamp'].diff(1)[1:].value_counts)
    
                        #time.sleep(round(time.time(),0)%20)
                        break
    
    
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    
            # model = pd.concat([train_old.copy(), train.copy()], ignore_index=True)
            # model = model.drop_duplicates(subset=['open_timestamp'])
            # model = model.tail(4000)
            # model.index = np.arange(len(model))


            if pre_train:

                pre_train = False
                
                train.insert(4, "change_int", 0, allow_duplicates=False)
                shifts = 12 * 15
                train.insert(5, f"Y_change15",train[f"change{shifts}"].shift(-shifts)*100)
    
                train.insert(6, "change15_int", 0, allow_duplicates=False)
                
    
                        
                while len(train) > 0 and train.iloc[-1]['open_timestamp'] % (3600*2) != 0:
                    train = train.iloc[:-1]
    
            
                train = train.copy()[train['open_timestamp'] % 5 == 0]

                # assert False
                for value in model_length:
                    
                    model = train.tail(value).copy()
                    model.index = np.arange(len(model))
                    model_list[f"model{value}"] = model
                    
                # model = train.tail(int(5760/12)).copy()
                # model.index = np.arange(len(model))
            
            
            
                # train_old = model.copy()

    
                
                
                for value in model_length:

                    model = model_list[f"model{value}"].copy()
                
                    model_path60 = model_list[f"model_path60_{value}"]
                    model_path15 = model_list[f"model_path15_{value}"]
                    
                    # Binning constants moved to ../feature_method.py (private).
                    model['change_int'] = bin_change_int_series(model['change'])
                
                    x_train = model.iloc[:, 14:]
                    y_train = model["change_int"]

                    print("shape:",model.shape,end = " ,")
                    print(model.columns[:15].tolist())
                    feature_columns = model.columns[14:].tolist()
                    target_column = "change_int"
                
                    # train_dataset = FinancialSequenceDataset(model, target_column, feature_columns, seq_len=60, scale_features=True)
                    train_dataset = FinancialSequenceDataset(model, target_column, feature_columns, seq_len=12,
                                    scale_features=True, scale_target=False
                                )
                    
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
                
                    LSTMmodel = TimeSeriesLSTM(
                        input_size=len(feature_columns), 
                        hidden_size=len(feature_columns), 
                        num_layers=4
                    )
                    

                    # trainer = FinancialTransformerTrainer(LSTMmodel)
                    trainer = FinancialTransformerTrainer(
                        model=LSTMmodel,
                        model_path=model_path60,
                        device='cuda'
                    )
                    
                    optimizer = optim.Adam(LSTMmodel.parameters(), lr=0.001)
                    
                    device='cuda' if torch.cuda.is_available() else 'cpu'
    
                    result = trainer.train_multiple_epochs(train_loader, optimizer, num_epochs=100,save_interval = 100)

                    model_list[f"train_dataset{value}"] = train_dataset
                    model_list[f"train_loader{value}"] = train_loader
                    model_list[f"LSTMmodel{value}"] = LSTMmodel
                    model_list[f"trainer{value}"] = trainer
                    model_list[f"optimizer{value}"] = optimizer
                    model_list[f"result{value}"] = result
                    
                    """Second model"""
                    
                    # Binning constants moved to ../feature_method.py (private).
                    model['change15_int'] = bin_change15_int_series(model['Y_change15'])
                    
                    feature_columns2 = columnB
                    target_column2 = "change15_int"
                    
                    # train_dataset2 = FinancialSequenceDataset(model, target_column, feature_columns, seq_len=12, scale_features=True)
                    train_dataset2 = FinancialSequenceDataset(model, target_column2, feature_columns2, seq_len=12,
                                    scale_features=True, scale_target=False
                                )
                    
                    train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=False)
                    
                    LSTMmodel2 = TimeSeriesLSTM(
                        input_size=len(feature_columns2), 
                        hidden_size=len(feature_columns2), 
                        num_layers=4
                    )
                    
                    # trainer2 = FinancialTransformerTrainer(LSTMmodel2)
                    trainer2 = FinancialTransformerTrainer(
                        model=LSTMmodel2,
                        model_path=model_path15,
                        device='cuda'
                    )
                    # result = trainer.train_multiple_epochs(
                    #     train_loader=train_loader,
                    #     optimizer=optimizer,
                    #     num_epochs=100,
                    #     save_dir='checkpoints',
                    #     save_interval=50
                    # )
                    
                    optimizer2 = optim.Adam(LSTMmodel2.parameters(), lr=0.001)
    
                    result2 = trainer2.train_multiple_epochs(train_loader2, optimizer2, num_epochs=100,save_interval=100)


                    model_list[f"train_dataset2{value}"] = train_dataset2
                    model_list[f"train_loader2{value}"] = train_loader2
                    model_list[f"LSTMmodel2{value}"] = LSTMmodel2
                    model_list[f"trainer2{value}"] = trainer2
                    model_list[f"optimizer2{value}"] = optimizer2
                    model_list[f"result2{value}"] = result2
                # result.trainer = trainer 

            # assert False
            
            lastest_model = model['open_date'].values[-1]
            # print('latest model:',lastest_model)
            # print(model['open_timestamp'].diff(1)[1:].value_counts())
      
            df4_old = pd.concat([df4_old.copy(), df4_temp.copy()], ignore_index=True)

            df4_old = df4_old.drop_duplicates(subset=['timestamp'])

            df4_old.index = np.arange(len(df4_old))

            lastest_model_open = int(model['open_timestamp'].values[-1])
            lastest_model_close = int(model['close_timestamp'].values[-1])
            
            # print('latest model:',lastest_model,'current time:',df4_old['snapshotTime'].values[-1],end = " ")
            # print(model['open_timestamp'].diff(1)[1:].value_counts())
            
            last_timestamp = int(df4_old['timestamp'].values[-1])
            
            last_open_time = model['open_timestamp'].values[-1]
            
            model_and_df8 = (last_timestamp - last_open_time)
            
            df_1min = df4_old.copy()

            close_price_1min = round(df_1min.close.values[-1],0)
            last_price_1min =  df_1min.close.values[-2]
            low_price_1min = df_1min.low.values[-2]
            max_price_1min = df_1min.high.values[-2]
            mid_price_1min = (low_price_1min + max_price_1min) / 2
    
            time_1min = df_1min.timestamp.values[-1]

            print('latest model:',lastest_model,'current time:',df4_old['snapshotTime'].values[-1],close_price_1min,end = " ")
            #raise ValueError("This is a custom error message.")
 
            total_df = df_1min


            close_series = total_df['close']
            # volume_series = total_df['volume']
            
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
            
            # Convert to DataFrame with proper column names
            changes_df = pd.DataFrame(
                changes,
                columns=[f'change{period}' for period in periods],
                index=total_df.index
            )
            
            total_df = pd.concat([total_df, changes_df], axis=1)

          
            total_df = total_df.dropna()
            
            total_df1 = total_df[total_df['timestamp'] % 5 == 0]
            total_df1.index = np.arange(len(total_df1))
        
            shifts = int(target/1)
            
            total_df1.insert(4,"change",total_df1[f"change{target}"].shift(-shifts)*100)
            total_df1.insert(5,"buy_sell","buy")
            total_df1.insert(6,"holding_time",3600 * 1)
            
            total_df1 = total_df1.drop(['high', 'low','open','open'], axis=1)
            
            total_df1.insert(6, "open_timestamp",total_df1.timestamp)
            total_df1.insert(7, "close_timestamp",total_df1.timestamp.shift(-int(shifts)))
            total_df1.insert(8, "open_date",total_df1.snapshotTime)
            total_df1.insert(9, "close_date",total_df1.snapshotTime.shift(-int(shifts)))
            total_df1["holding_time"] = total_df1["close_timestamp"] - total_df1["open_timestamp"]
            
            total_df1.insert(5,"Y_change15",total_df1[f"change{12*15}"].shift(-shifts)*100)
            
            print('model spread:',model_and_df8)

            # assert False
            # raise ValueError("Stop !")
            # is_buy = False

            
            if not is_buy:
                total_df1['change'] = -total_df1['change']
                total_df1['buy_sell'] = "sell"

                
            # if model_and_df8 >= (3600*2 + 60 * target):
            
            if model_and_df8 >= (3600*3):   
                # print('model spread:',model_and_df8)
                
                print('pre_train is set True')

                pre_train = True

                check_buy = 0
            
            # elif (3600*1 + 60 * target) >= model_and_df8 > 60 * target:
            # elif model_and_df8 >= 60 * target:
            elif model_and_df8 >= 3600:   
                # print('model spread:',model_and_df8)
                # test_since = int(lastest_model_open-(3600/5))
                # vaild_df = total_df1.query(f"open_timestamp > {test_since}").copy()
                vaild_df = total_df1.tail(2).copy()
                vaild_df.loc[:, 'change'] = 0  # ✅ Safe and explicit
                # for epochA in [100,125,150]:
                for epochA in [100]:
                    

                    symbol = 'BTCUSD'
                    order_size = 0.01
                    MIN_TIME_BETWEEN_ORDERS = 300
                    
                    if stop_market() and is_buy:


                        for value in model_length:
        
                            check_buy = model_list[f"trainer{value}"].predict(test_df=vaild_df,feature_columns_start=12,target_column='change',
                                                                              feature_scaler=model_list[f"train_dataset{value}"].feature_scaler )
                            check_buy2 = model_list[f"trainer2{value}"].predict(test_df=vaild_df[columnB],feature_columns_start=0,target_column='change180',
                                                                                feature_scaler=model_list[f"train_dataset2{value}"].feature_scaler )
                            
                            print("model:",value,check_buy,check_buy2, vaild_df.snapshotTime.values[-1],vaild_df.close.values[-1])
        
                            if check_buy >= 0.70 and check_buy2 >= 0.70 :
                                insert_signal(DB_CONFIG, BUY_SIGNAL_TABLE, symbol, check_buy, check_buy2, vaild_df.open_timestamp.values[-1])

                                try:
                                    # TIMESTAMP-BASED DUPLICATION CHECK
                                    current_time = time.time()
                                    
                                    # Check if we have any recent orders for this coin
                                    recent_order_found = False
                                    if order_list["start_time"]:
                                        # Get the most recent order time
                                        latest_order_time = max(order_list["start_time"])
                                        time_since_last_order = current_time - latest_order_time + 3600 * 8
                                        
                                        if time_since_last_order < MIN_TIME_BETWEEN_ORDERS:
                                            print(f"跳过开仓: 距离上次订单仅 {time_since_last_order:.1f} 秒, "
                                                  f"需要等待 {MIN_TIME_BETWEEN_ORDERS} 秒")
                                            recent_order_found = True
                                            
                                    if not recent_order_found:
                                        
        
                                        trade_try = 1
                                        while trade_try > 0:
                                            try:

                                                if connect_to_mt5():

                                                    trade_result = buy_order(symbol,order_size)
                                                    oid = trade_result.order
                                                    print(f"开仓成功! 订单ID: {oid}") 
                                                    order_list["order_ID"].append(oid)
                                                    order_list["start_time"].append(time_1min + 3600 * 8)
        
                                                    print(f"订单已记录 - OID: {oid}, 时间: {time_1min}")
                                                    
                                                    break
                                                else:
                                                    # time.sleep(5)
                                           
                                                    continue
                                                    
                                            except AttributeError as e:
                                                trade_try -= 1
                                                print(e)
                                        
                            
                                        else:
                                            print(f"开仓失败: {trade_result}")
                                    
                                except Exception as e:
                                    print(f"开仓过程出错: {e}")

                    elif stop_market() and not is_buy:

                        check_sell = trainer.predict(test_df=vaild_df,feature_columns_start=12,target_column='change',feature_scaler=train_dataset.feature_scaler )
                        check_sell2 = trainer2.predict(test_df=vaild_df[columnB],feature_columns_start=0,target_column='change15',feature_scaler=train_dataset2.feature_scaler )
                        
                        print(f"epochA:{epochA},sell:",check_sell,"sell2:",check_sell2 ,",720-2160:", vaild_df.shape, close_price_1min ,end = " ")
                        print("last vaild_df:",vaild_df.open_date.values[-1])
    
                        if check_sell >= 0.21 and check_sell2 >= 0.41 :
                            insert_signal(DB_CONFIG, SELL_SIGNAL_TABLE, symbol, check_sell, check_sell2, vaild_df.open_timestamp.values[-1])

                            try:
                                # TIMESTAMP-BASED DUPLICATION CHECK
                                current_time = time.time()
                                
                                # Check if we have any recent orders for this coin
                                recent_order_found = False
                                if order_list["start_time"]:
                                    # Get the most recent order time
                                    latest_order_time = max(order_list["start_time"])
                                    time_since_last_order = current_time - latest_order_time + 3600 * 8
                                    
                                    if time_since_last_order < MIN_TIME_BETWEEN_ORDERS:
                                        print(f"跳过开仓: 距离上次订单仅 {time_since_last_order:.1f} 秒, "
                                              f"需要等待 {MIN_TIME_BETWEEN_ORDERS} 秒")
                                        recent_order_found = True
                                
                                if not recent_order_found:
                                    
    
                                    trade_try = 1
                                    while trade_try > 0:
                                        try:

                                            if connect_to_mt5():

                                                trade_result = sell_order(symbol,order_size)
                                                oid = trade_result.order
                                                print(f"开仓成功! 订单ID: {oid}") 
                                                order_list["order_ID"].append(oid)
                                                order_list["start_time"].append(time_1min + 3600 * 8)
    
                                                print(f"订单已记录 - OID: {oid}, 时间: {time_1min}")
                                                
                                                break
                                            else:
                                                # time.sleep(5)
                                       
                                                continue
                                                
                                        except AttributeError as e:
                                            trade_try -= 1
                                            print(e)
                                    
                        
                                    else:
                                        print(f"开仓失败: {trade_result}")
                                
                            except Exception as e:
                                print(f"开仓过程出错: {e}")
        
            
                    
                    
                            # Optionally add error logging here
            # MAX_HOLDING_TIME = 60 * target
            MAX_HOLDING_TIME = 3600
            # latest_order_time = max(order_list["start_time"])
            # holding_time = current_time - latest_order_time + 3600 * 8
            
         
                # buy_positions = [pos for pos in mt5.positions_get() if pos.type == mt5.ORDER_TYPE_BUY]
            positions = []
            if mt5 is not None:
                try:
                    positions = mt5.positions_get()
                except Exception as e:
                    print(f"MT5 positions_get error: {e}")
            if positions is None:
                positions = []
            if len(positions) > 0:
                for position in positions:
                    
                    if is_buy and "BUY" in position.comment:
                        holding_time = time.time()  - position.time + 3600 * 3
                        if holding_time >= MAX_HOLDING_TIME:
                            close_position_by_ticket(ticket=position.ticket)
                            
                    elif not is_buy and "SELL" in position.comment:
                            
                        fholding_time = time.time()  - position.time + 3600 * 3
                        if holding_time >= MAX_HOLDING_TIME:
                            close_position_by_ticket(ticket=position.ticket)
                    
                        

            
                    
            now = time.time()
            cost = round((now- start_time),2)
            # print('cost:',cost, end = " ")
            # # now = time.time()
            
            sleep_sec = 6 - (start_time % 5)

            if cost > sleep_sec:
                print('cost:',cost, "sleep_sec:",0)
                pass
            else:
                real_sleep = round((sleep_sec-cost),2) 
                print('cost:',cost,"sleep_sec:",real_sleep)
                time.sleep(real_sleep)

            print()

            # assert False
            # # print('cost:',cost)
            # # if cost > 65:
            #     pass
            # else:
            #     time.sleep(65-cost)

            #     sleep_sec
            # print('cost:',cost)
                
            #break
            
#thread2 = threading.Thread(target=thread_function2)
#thread2.start()


