# QuantTrading — BTCUSD Quantitative Trading System

A production-oriented algorithmic trading system for **BTCUSDT** that pairs a
Python research & training pipeline with a real-time **Rust execution bot**.
It collects 5-second Binance klines, engineers period-based change features,
trains LSTM models on GPU, and generates buy/sell signals — with MetaTrader 5
as the broker interface.

```
Binance (5s klines) ──► TimescaleDB ──► Feature Engineering ──► LSTM Inference (GPU) ──► Buy/Sell Signals ──► MT5 Execution
```

## Components

### 1. Data Pipeline (Python)

Collects and maintains market data:

- Fetches Binance klines and resamples them into 5-second OHLCV bars
- Writes into a TimescaleDB hypertable with continuous aggregates
- Backfills gaps in raw and training data (`backfill_raw_gap.py`,
  `backfill_training_gaps.py`)
- Migrates legacy MySQL data into TimescaleDB and fixes schema/timezone
  issues (`migrate_*.py`, `fix_*.py`)

### 2. Feature Engineering & Model Training (Python)

The quant research side:

- Computes period-based percentage-change features from close prices
  (12s → 24h lookback windows, 336 features)
- Builds buy/sell training rows with a 1-hour forward target
- Trains LSTM sequence models on GPU with discrete change bins as targets
  (`BTCUSD_LSTM_part3_336_5s_singal.py`, `BTCUSD_all_in_one*.py`)
- Saves scalers and exports TorchScript models for Rust inference

### 3. Signal Generation & Trading (Python + Rust)

- The Python scripts generate signals from the trained models and insert
  them into signal tables
- `BTCUSD_all_in_one*_MT5.py` reads those signals and executes trades via
  MetaTrader 5 (orders, positions, close logic)
- The Rust bot (`Rust/price_newDB.rs`) runs the same live loop every 5
  seconds: fetch klines → compute features → GPU inference (libtorch via
  `tch`) → write buy/sell training rows and signal rows

### 4. Configuration & Secrets (`env.txt` — private)

All credentials and trading parameters live in a single `env.txt` at the
repository root (outside `Python/` and `Rust/`):

- Database connection (host, port, user, password, tables)
- Model and scaler paths, signal thresholds, spread limits
- MySQL and MetaTrader 5 credentials

`Python/db_env.py` loads these values for every script; the Rust bot reads
the same file at startup.

## Repository Layout

```
QuantTrading/
├── Python/            # Data pipeline, feature engineering, LSTM training, signals, MT5 execution
│   ├── db_env.py      # Shared env.txt loader (credentials + config)
│   ├── price_newDB.py # Async data ingestion + training-data builder
│   ├── BTCUSD_*.py    # Training / signal / trade-execution scripts
│   └── migrate_*.py   # MySQL → TimescaleDB migrations
├── Rust/              # Real-time 5s bot (tokio, sqlx, tch/libtorch)
│   ├── price_newDB.rs # Main bot: ingestion, features, GPU inference, pre-train
│   ├── Cargo.toml     # Cargo manifest
│   └── procedure.txt  # Build & run instructions
└── env.txt            # PRIVATE — not committed
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Research & training | Python 3.11, pandas, numpy, PyTorch, scikit-learn, asyncio |
| Live execution bot | Rust, tokio, reqwest, sqlx, tch (libtorch / CUDA 12) |
| Databases | TimescaleDB / PostgreSQL (primary), MySQL (legacy source) |
| Data source | Binance public kline API (5s / 1m) |
| Broker interface | MetaTrader 5 |

## Privacy Notice

This is a **public** repository. Intentionally **not** included:

- `env.txt` — database credentials, MT5 account, and all trading parameters
- `feature_method.py` / `feature_method.rs` — the proprietary feature
  computation method
- `*.npy` model-selection data, trained model checkpoints, and scalers

These files stay private and are read from the repository root at runtime.

## Disclaimer

For research and educational purposes only. Not financial advice. Trading
cryptocurrency involves substantial risk.
