# QuantTrading — BTCUSD Quantitative Trading System

A production-oriented algorithmic trading system for **BTCUSDT** that pairs a
Python research & training pipeline with a real-time **Rust execution bot**.
It collects 5-second Binance klines, engineers period-based change features,
trains LSTM models on GPU, and trades via signals — with MetaTrader 5 as the
broker interface.

```
Binance (5s klines) ──► TimescaleDB ──► Feature Engineering ──► LSTM Inference (GPU) ──► Buy/Sell Signals ──► MT5 Execution
```

## Repository Layout

| Path | Purpose |
|------|---------|
| `Python/` | Data collection, backfills, DB migrations, feature engineering, LSTM training (PyTorch), signal generation, MT5 trade execution |
| `Rust/` | Real-time 5-second bot: Binance → TimescaleDB → feature computation → GPU inference (`tch-rs` + TorchScript) → training rows & signals |

### Key scripts (`Python/`)

- `price_newDB.py` — async data pipeline: 5s OHLCV ingestion, feature computation, buy/sell training data
- `BTCUSD_LSTM_part3_336_5s_singal.py` — LSTM model training + signal generation
- `BTCUSD_all_in_one*.py` — combined pipeline and its MT5 execution counterpart
- `migrate_*.py` — MySQL → TimescaleDB migrations and schema fixes
- `db_env.py` — loads all credentials/config from the root `env.txt`

### Rust bot (`Rust/`)

- `price_newDB.rs` — ~1,500-line async bot: 5s kline polling, TimescaleDB (sqlx),
  feature computation, CUDA inference via libtorch (`tch`), pre-train loop,
  buy/sell row insertion and signal writing

## Tech Stack

- **Python 3.11** — asyncio, asyncpg, pandas, numpy, PyTorch, scikit-learn, MetaTrader5
- **Rust** — tokio, reqwest, sqlx, tch/libtorch (CUDA 12), serde
- **Database** — TimescaleDB/PostgreSQL (primary), MySQL (legacy source)
- **Data source** — Binance public kline API (5s/1m)

## Author — Quant Trader & Data Engineer

I'm a quant trader and data engineer who builds algorithmic trading systems
end-to-end: from raw market data ingestion and feature engineering, through
ML model training and backtesting, to low-latency execution and live
monitoring.

What I enjoy most is the intersection of the two disciplines — turning messy
high-frequency market data into reliable features, then engineering the
systems fast enough to act on them in real time. This repository is a good
example of that mix: the research and training side lives in Python
(pandas/numpy/PyTorch), while the live 5-second loop is written in Rust with
GPU inference to keep the pipeline fast and stable.

I work with the full stack of a trading system:

- **Data engineering** — streaming ingestion, TimescaleDB/PostgreSQL schema
  design, time-series migrations, data validation and gap backfilling
- **Quant research** — feature engineering, LSTM sequence models, signal
  threshold calibration, model selection from historical performance
- **Systems / execution** — async pipelines, GPU-accelerated inference,
  broker integration (MetaTrader 5), and long-running production processes

## Privacy Notice

This is a **public** repository. The following are intentionally **not**
included:

- `env.txt` — database credentials, MT5 account, and all trading parameters
- `feature_method.py` / `feature_method.rs` — the proprietary feature
  computation method
- `*.npy` model-selection data, trained model checkpoints, and scalers

These files stay private on the author's machine; the public code reads them
from the repository root at runtime.

## Disclaimer

For research and educational purposes only. Not financial advice. Trading
cryptocurrency involves substantial risk.
