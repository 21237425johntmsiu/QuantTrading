// price_newDB.rs — Rust rewrite of price_newDB.py
// Binance 1s klines → TimescaleDB → feature engineering → buy/sell training data

use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::Client;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tch::{CModule, Device, Tensor, Kind};
use tch::nn::{self, Module, OptimizerConfig};
use tch::nn::RNN;
use std::io::Write;

// Proprietary feature computation lives OUTSIDE this crate at the repo root
// (feature_method.rs) — PRIVATE, never committed to the public repo.
// Call it as feature_method::...
#[path = "../feature_method.rs"]
mod feature_method;

// ──────────────────── Configuration ────────────────────

static CONFIG: OnceLock<Config> = OnceLock::new();
static PRE_TRAIN_DONE: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
struct Config {
    db_host: String, db_port: u16, db_user: String, db_password: String, db_name: String,
    db_min_size: u32, db_max_size: u32,
    raw_table: String, buy_table: String, sell_table: String,
    buy_signal_table: String, sell_signal_table: String,
    symbol: String, semaphore_limit: usize,
    buy_threshold: f64, buy_threshold2: f64, sell_threshold: f64, sell_threshold2: f64,
    model1_path: String, model2_path: String,
    scaler1_mean_path: String, scaler1_scale_path: String,
    scaler2_mean_path: String, scaler2_scale_path: String,
    signal_trim_rows: i64,
    spread_target: i64, spread_retrain: i64,
    pre_train: bool,
}

impl Config {
    fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut map = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some((key, value)) = line.split_once('=') {
                map.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
        fn req(map: &HashMap<String, String>, key: &str) -> String {
            map.get(key).cloned().unwrap_or_else(|| panic!("env.txt missing key: {}", key))
        }
        Ok(Config {
            db_host: req(&map, "DB_HOST"),
            db_port: req(&map, "DB_PORT").parse()?,
            db_user: req(&map, "DB_USER"),
            db_password: req(&map, "DB_PASSWORD"),
            db_name: req(&map, "DB_NAME"),
            db_min_size: req(&map, "DB_MIN_SIZE").parse()?,
            db_max_size: req(&map, "DB_MAX_SIZE").parse()?,
            raw_table: req(&map, "RAW_TABLE"),
            buy_table: req(&map, "BUY_TABLE"),
            sell_table: req(&map, "SELL_TABLE"),
            buy_signal_table: req(&map, "BUY_SIGNAL_TABLE"),
            sell_signal_table: req(&map, "SELL_SIGNAL_TABLE"),
            symbol: req(&map, "SYMBOL"),
            semaphore_limit: req(&map, "SEMAPHORE_LIMIT").parse()?,
            buy_threshold: req(&map, "BUY_THRESHOLD").parse()?,
            buy_threshold2: req(&map, "BUY_THRESHOLD2").parse()?,
            sell_threshold: req(&map, "SELL_THRESHOLD").parse()?,
            sell_threshold2: req(&map, "SELL_THRESHOLD2").parse()?,
            model1_path: req(&map, "MODEL1_PATH"),
            model2_path: req(&map, "MODEL2_PATH"),
            scaler1_mean_path: req(&map, "SCALER1_MEAN_PATH"),
            scaler1_scale_path: req(&map, "SCALER1_SCALE_PATH"),
            scaler2_mean_path: req(&map, "SCALER2_MEAN_PATH"),
            scaler2_scale_path: req(&map, "SCALER2_SCALE_PATH"),
            signal_trim_rows: req(&map, "SIGNAL_TRIM_ROWS").parse()?,
            spread_target: req(&map, "SPREAD_TARGET").parse()?,
            spread_retrain: req(&map, "SPREAD_RETRAIN").parse()?,
            pre_train: req(&map, "PRE_TRAIN") == "true",
        })
    }
}

// ──────────────────── Types ────────────────────

/// A single 5-second OHLCV record from either Binance or the DB.
#[derive(Debug, Clone)]
struct Record5s {
    /// HKT display time (naive, no timezone — matched to Python's snapshotTime)
    snapshot_time: NaiveDateTime,
    /// Unix epoch seconds (UTC)
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// A row with computed percentage-change features and a forward target change.
#[derive(Debug, Clone)]
struct FeatureRow {
    snapshot_time: NaiveDateTime,
    timestamp: i64,
    close: f64,
    volume: f64,
    /// Percentage changes at each period: changes[j] = change over periods[j]
    changes: Vec<f64>,
}

/// A row destined for the buy or sell training table.
#[derive(Debug, Clone)]
struct TrainingRow {
    snapshot_time: NaiveDateTime,
    timestamp: i64,
    close: f64,
    volume: f64,
    /// Forward-looking target change (scaled)
    change: f64,
    buy_sell: String,
    holding_time: i64,
    open_timestamp: i64,
    close_timestamp: i64,
    open_date: NaiveDateTime,
    close_date: NaiveDateTime,
    changes: Vec<f64>,
}

// ──────────────────── Helpers ────────────────────

/// Convert a UTC `DateTime` to a naive HKT (UTC+8) datetime for display.
fn utc_to_hkt_naive(dt: DateTime<Utc>) -> NaiveDateTime {
    let hkt_offset = chrono::FixedOffset::east_opt(8 * 3600).unwrap();
    dt.with_timezone(&hkt_offset).naive_local()
}

/// Round an f64 to `decimals` places (like np.round(…, 4)).
fn round_to(value: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    (value * factor).round() / factor
}

/// Load a CSV file of comma-separated f64 values into a Vec<f64>.
fn load_csv_f64(path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in content.lines() {
        for token in line.split(',') {
            let t = token.trim();
            if !t.is_empty() {
                values.push(t.parse::<f64>()?);
            }
        }
    }
    Ok(values)
}

/// A TorchScript model loaded via tch-rs for LSTM inference on GPU.
struct LSTMModel {
    model: CModule,
    device: Device,
}

impl LSTMModel {
    /// Load a TorchScript `.pt` file onto the specified device (CPU or CUDA).
    fn load(path: &str, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        eprintln!("Loading model from {}", path);
        let model = CModule::load_on_device(path, device)?;
        Ok(LSTMModel { model, device })
    }

    /// Run inference on a single row of features (standardized).
    /// `features`: slice of standardized feature values (len = n_features).
    /// `seq_len`: sequence length expected by the model (default 12).
    fn predict(&self, features: &[f64], seq_len: i64) -> Result<f64, Box<dyn std::error::Error>> {
        // Input shape: (1, seq_len, n_features). Model expects float32.
        let n_features = features.len() as i64;
        let float_features: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        let input_data: Vec<f32> = float_features.iter().copied().cycle().take((seq_len * n_features) as usize).collect();
        let input = Tensor::from_slice(&input_data)
            .to_device(self.device)
            .view([1, seq_len, n_features]);

        let output = self.model.forward_ts(&[input])?;
        let scalar = output.double_value(&[0]);
        Ok(scalar)
    }
}

/// Standardize features using pre-computed mean and scale.
fn standardize(features: &[f64], mean: &[f64], scale: &[f64]) -> Vec<f64> {
    features.iter()
        .zip(mean.iter())
        .zip(scale.iter())
        .map(|((&f, &m), &s)| if s == 0.0 { 0.0 } else { (f - m) / s })
        .collect()
}

// ──────────────────── DB Initialization ────────────────────

async fn init_db() -> Result<PgPool, sqlx::Error> {
    let cfg = CONFIG.get().unwrap();
    let pool = PgPoolOptions::new()
        .min_connections(cfg.db_min_size)
        .max_connections(cfg.db_max_size)
        .connect(&format!(
            "postgres://{}:{}@{}:{}/{}",
            cfg.db_user, cfg.db_password, cfg.db_host, cfg.db_port, cfg.db_name
        ))
        .await?;

    // Enable TimescaleDB extension
    sqlx::query("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        .execute(&pool)
        .await?;

    // Create raw 5s table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS btcusd_5s (
            "timestamp" TIMESTAMPTZ NOT NULL,
            open FLOAT8 NOT NULL,
            high FLOAT8 NOT NULL,
            low FLOAT8 NOT NULL,
            close FLOAT8 NOT NULL,
            volume FLOAT8 NOT NULL
        );
        "#,
    )
    .execute(&pool)
    .await?;

    // Convert to hypertable if not already
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM _timescaledb_catalog.hypertable WHERE table_name = 'btcusd_5s')",
    )
    .fetch_one(&pool)
    .await?;

    if !exists {
        sqlx::query(
            "SELECT create_hypertable('btcusd_5s', 'timestamp', chunk_time_interval => INTERVAL '1 day');",
        )
        .execute(&pool)
        .await?;
        eprintln!("Created hypertable btcusd_5s");
    }

    // Unique index for ON CONFLICT DO NOTHING
    sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS idx_btcusd_5s_time ON btcusd_5s (\"timestamp\");")
        .execute(&pool)
        .await?;

    // Compression for chunks older than 7 days
    let _ = sqlx::query("ALTER TABLE btcusd_5s SET (timescaledb.compress);")
        .execute(&pool)
        .await;
    let _ = sqlx::query(
        "SELECT add_compression_policy('btcusd_5s', INTERVAL '7 days', if_not_exists => TRUE);",
    )
    .execute(&pool)
    .await;

    // Continuous aggregate for 1-minute resampling
    sqlx::query(
        r#"
        CREATE MATERIALIZED VIEW IF NOT EXISTS btcusd_1m
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 min', "timestamp") AS bucket,
            first(open, "timestamp") AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, "timestamp") AS close,
            sum(volume) AS volume
        FROM btcusd_5s
        GROUP BY bucket;
        "#,
    )
    .execute(&pool)
    .await?;

    let _ = sqlx::query(
        r#"
        SELECT add_continuous_aggregate_policy('btcusd_1m',
            start_offset => INTERVAL '1 day',
            end_offset => INTERVAL '1 min',
            schedule_interval => INTERVAL '1 minute');
        "#,
    )
    .execute(&pool)
    .await;

    // Signal tables for buy/sell predictions
    let cfg = CONFIG.get().unwrap();
    let signal_tables = [&*cfg.buy_signal_table, &*cfg.sell_signal_table];
    for table_name in &signal_tables {
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {} (symbol TEXT, close FLOAT8, check_buy FLOAT8, check_buy2 FLOAT8, check_buy3 FLOAT8 DEFAULT 0, open_timestamp BIGINT);",
            table_name
        );
        let _ = sqlx::query(&sql).execute(&pool).await;
        // Add columns if table already existed without them
        for (col, col_type) in [("check_buy3", "FLOAT8 DEFAULT 0"), ("close", "FLOAT8")] {
            let alter_sql = format!(
                "ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {};",
                table_name, col, col_type
            );
            let _ = sqlx::query(&alter_sql).execute(&pool).await;
        }
    }

    Ok(pool)
}

// ──────────────────── Binance API ────────────────────

/// Fetch 1s kline data from Binance. Returns parsed kline arrays.
///
/// Binance returns JSON arrays of arrays. We parse each inner array as a
/// 12-element `Vec<serde_json::Value>`.
async fn fetch_klines(
    client: &Client,
    symbol: &str,
    interval: &str,
    start_time_ms: i64,
    limit: u32,
) -> Result<Vec<[serde_json::Value; 12]>, Box<dyn std::error::Error>> {
    let url = "https://api.binance.com/api/v3/klines";
    let resp = client
        .get(url)
        .query(&[
            ("symbol", symbol),
            ("interval", interval),
            ("startTime", &start_time_ms.to_string()),
            ("limit", &limit.to_string()),
        ])
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await?;
        return Err(format!("Binance API error: {}", text).into());
    }

    let raw: Vec<Vec<serde_json::Value>> = resp.json().await?;
    let mut out = Vec::with_capacity(raw.len());

    for arr in raw {
        if arr.len() < 12 {
            continue;
        }
        let mut fixed: [serde_json::Value; 12] = Default::default();
        for (i, v) in arr.into_iter().enumerate().take(12) {
            fixed[i] = v;
        }
        out.push(fixed);
    }

    Ok(out)
}

// ──────────────────── Resampling ────────────────────

/// Resample 1-second klines to N-second OHLCV.  Returns a `Vec<Record5s>`.
///
/// Equivalent to Python's `df_reshape_1s`.
fn resample_to_n_seconds(raw: &[[serde_json::Value; 12]], length: i64) -> Vec<Record5s> {
    if raw.is_empty() {
        return vec![];
    }

    // Parse each kline into a simpler struct
    struct Kline1s {
        start_ms: i64, // ms
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    }

    let klines: Vec<Kline1s> = raw
        .iter()
        .filter_map(|k| {
            Some(Kline1s {
                start_ms: k[0].as_i64()?,
                open: k[1].as_str()?.parse().ok()?,
                high: k[2].as_str()?.parse().ok()?,
                low: k[3].as_str()?.parse().ok()?,
                close: k[4].as_str()?.parse().ok()?,
                volume: k[5].as_str()?.parse().ok()?,
            })
        })
        .collect();

    if klines.is_empty() {
        return vec![];
    }

    // Group by bucket: (start_ms / (length * 1000)) * (length * 1000)
    let bucket_ms = length * 1000;

    // We'll collect into a BTreeMap keyed by bucket start (ms)
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<i64, Vec<&Kline1s>> = BTreeMap::new();

    for k in &klines {
        let bucket = (k.start_ms / bucket_ms) * bucket_ms;
        buckets.entry(bucket).or_default().push(k);
    }

    let mut out = Vec::with_capacity(buckets.len());
    for (&bucket_start_ms, group) in &buckets {
        let bucket_start_s = bucket_start_ms / 1000;
        let open = group[0].open;
        let mut high = group[0].high;
        let mut low = group[0].low;
        let close = group.last().unwrap().close;
        let mut volume = 0.0;

        for k in group {
            if k.high > high {
                high = k.high;
            }
            if k.low < low {
                low = k.low;
            }
            volume += k.volume;
        }

        // snapshotTime = HKT display of bucket start
        let hkt_dt =
            DateTime::from_timestamp(bucket_start_s + 8 * 3600, 0)
                .map(|dt| dt.naive_utc())
                .unwrap_or_default();

        out.push(Record5s {
            snapshot_time: hkt_dt,
            timestamp: bucket_start_s,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    out
}

// ──────────────────── Feature Engineering ────────────────────
// compute_features / prepare_train_data / compute_periods /
// compute_column_b_periods / binning moved to ../feature_method.rs
// (private, outside Rust/).

// ──────────────────── DB Operations ────────────────────

/// Insert 5s OHLCV rows into the hypertable. Returns count of inserted rows.
async fn insert_5s_batch(pool: &PgPool, records: &[Record5s]) -> Result<usize, sqlx::Error> {
    if records.is_empty() {
        return Ok(0);
    }

    let mut builder = sqlx::QueryBuilder::new(
        "INSERT INTO btcusd_5s (\"timestamp\", open, high, low, close, volume) ",
    );
    builder.push_values(records, |mut b, r| {
        let utc_dt = DateTime::from_timestamp(r.timestamp, 0).unwrap();
        b.push_bind(utc_dt)
            .push_bind(r.open)
            .push_bind(r.high)
            .push_bind(r.low)
            .push_bind(r.close)
            .push_bind(r.volume);
    });
    builder.push(" ON CONFLICT (\"timestamp\") DO NOTHING");
    let result = builder.build().execute(pool).await?;
    Ok(result.rows_affected() as usize)
}

/// Create a training table with the known schema + dynamic change columns.
async fn create_training_table(
    pool: &PgPool,
    table_name: &str,
    periods: &[usize],
) -> Result<(), sqlx::Error> {
    let mut col_defs = vec![
        r#""snapshotTime" TIMESTAMP"#.to_string(),
        r#""timestamp" BIGINT"#.to_string(),
        r#""close" FLOAT8"#.to_string(),
        r#""volume" FLOAT8"#.to_string(),
        r#""change" FLOAT8"#.to_string(),
        r#""buy_sell" TEXT"#.to_string(),
        r#""holding_time" BIGINT"#.to_string(),
        r#""open_timestamp" BIGINT"#.to_string(),
        r#""close_timestamp" BIGINT"#.to_string(),
        r#""open_date" TIMESTAMP"#.to_string(),
        r#""close_date" TIMESTAMP"#.to_string(),
    ];

    for &p in periods {
        col_defs.push(format!(r#""change{}" FLOAT8"#, p));
    }

    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {} ({});",
        table_name,
        col_defs.join(", ")
    );
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

/// Insert training data rows into a buy/sell table, creating the table if needed.
/// Batches in chunks to stay under PostgreSQL's u16::MAX parameter limit.
async fn insert_train_batch(
    pool: &PgPool,
    rows: &[TrainingRow],
    periods: &[usize],
    table_name: &str,
) -> Result<(), sqlx::Error> {
    if rows.is_empty() {
        return Ok(());
    }

    // Check if table exists
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
    )
    .bind(table_name)
    .fetch_one(pool)
    .await?;

    if !exists {
        create_training_table(pool, table_name, periods).await?;
    }

    // Build column list
    let fixed_cols = [
        "snapshotTime", "timestamp", "close", "volume", "change",
        "buy_sell", "holding_time", "open_timestamp", "close_timestamp",
        "open_date", "close_date",
    ];
    let change_cols: Vec<String> = periods.iter().map(|p| format!("change{}", p)).collect();
    let all_cols: Vec<&str> = fixed_cols
        .iter()
        .map(|c| *c)
        .chain(change_cols.iter().map(|c| c.as_str()))
        .collect();

    let cols_str = all_cols
        .iter()
        .map(|c| format!("\"{}\"", c))
        .collect::<Vec<_>>()
        .join(", ");

    let params_per_row = fixed_cols.len() + periods.len(); // 11 + 336 = 347
    let max_rows_per_batch = (u16::MAX as usize) / params_per_row;

    for chunk in rows.chunks(max_rows_per_batch) {
        let mut builder = sqlx::QueryBuilder::new(format!(
            "INSERT INTO {} ({}) ",
            table_name, cols_str
        ));

        builder.push_values(chunk, |mut b, r| {
            b.push_bind(r.snapshot_time)
                .push_bind(r.timestamp)
                .push_bind(r.close)
                .push_bind(r.volume)
                .push_bind(r.change)
                .push_bind(&r.buy_sell)
                .push_bind(r.holding_time)
                .push_bind(r.open_timestamp)
                .push_bind(r.close_timestamp)
                .push_bind(r.open_date)
                .push_bind(r.close_date);
            for &ch in &r.changes {
                b.push_bind(ch);
            }
        });
        // Skip rows that already exist (unique constraint on open_timestamp)
        builder.push(" ON CONFLICT (open_timestamp) DO NOTHING");

        builder.build().execute(pool).await?;
    }

    Ok(())
}

/// Load initial OHLCV data from the hypertable.
/// Equivalent to Python's `load_initial_data`.
async fn load_data(pool: &PgPool, query: &str) -> Result<Vec<Record5s>, sqlx::Error> {
    let rows = sqlx::query(query).fetch_all(pool).await?;

    if rows.is_empty() {
        return Ok(vec![]);
    }

    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        // Column names from query: "snapshottime" (lowercased by PG),
        // "timestamp", "open", "high", "low", "close", "volume"
        let time_utc: DateTime<Utc> = row.get("snapshottime");
        let ts: i64 = row.get("timestamp");
        let open: f64 = row.get("open");
        let high: f64 = row.get("high");
        let low: f64 = row.get("low");
        let close: f64 = row.get("close");
        let volume: f64 = row.get("volume");

        // Convert UTC to HKT naive for snapshot_time
        let hkt_naive = utc_to_hkt_naive(time_utc);

        out.push(Record5s {
            snapshot_time: hkt_naive,
            timestamp: ts,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    Ok(out)
}

/// Fetch the last `open_timestamp` from a training table.
async fn get_last_open(
    pool: &PgPool,
    table_name: &str,
) -> Result<(i64, Option<NaiveDateTime>), sqlx::Error> {
    let sql = format!(
        "SELECT open_timestamp FROM {} ORDER BY open_timestamp DESC LIMIT 1",
        table_name
    );
    let ts: Option<i64> = sqlx::query_scalar(&sql).fetch_optional(pool).await?;
    Ok((ts.unwrap_or(0), None))
}

/// Insert a signal into the signal table and trim to max_rows.
async fn insert_signal(
    pool: &PgPool,
    table_name: &str,
    symbol: &str,
    close: f64,
    check_buy: f64,
    check_buy2: f64,
    open_ts: i64,
    check_buy3: f64,
) -> Result<(), sqlx::Error> {
    let sql = format!(
        "INSERT INTO {} (symbol, close, check_buy, check_buy2, check_buy3, open_timestamp) VALUES ($1, $2, $3, $4, $5, $6)",
        table_name
    );
    sqlx::query(&sql)
        .bind(symbol)
        .bind(close)
        .bind(check_buy)
        .bind(check_buy2)
        .bind(check_buy3)
        .bind(open_ts)
        .execute(pool)
        .await?;
    trim_signal_table(pool, table_name).await;
    Ok(())
}

/// Trim signal table to keep only the latest `max_rows` rows.
async fn trim_signal_table(pool: &PgPool, table_name: &str) {
    let max_rows = CONFIG.get().unwrap().signal_trim_rows;
    let sql = format!(
        "DELETE FROM {}
         WHERE open_timestamp < (
             SELECT MIN(open_timestamp) FROM (
                 SELECT open_timestamp FROM {} ORDER BY open_timestamp DESC LIMIT {}
             ) AS latest
         );",
        table_name, table_name, max_rows
    );
    let _ = sqlx::query(&sql).execute(pool).await;
}

/// Trim training rows that trail beyond the last 2-hour-aligned open_timestamp.
/// Matches Python: while last row's open_timestamp % 7200 != 0, drop it.
async fn trim_to_aligned(pool: &PgPool, table_name: &str) {
    loop {
        let last_ts: Option<i64> = sqlx::query_scalar(&format!(
            "SELECT open_timestamp FROM {} ORDER BY open_timestamp DESC LIMIT 1", table_name
        )).fetch_optional(pool).await.unwrap_or(None);
        match last_ts {
            Some(ts) if ts > 0 && ts % 7200 != 0 => {
                let _ = sqlx::query(&format!(
                    "DELETE FROM {} WHERE open_timestamp = $1", table_name
                )).bind(ts).execute(pool).await;
            }
            _ => break,
        }
    }
}

// ──────────────────── Pre-Train Model Training ────────────────────

/// A row of training data fetched from a training table.
#[derive(Debug, Clone)]
struct TrainingDataRow {
    open_timestamp: i64,
    change: f64,
    changes: Vec<f64>,
}

/// Fetch pre-training data from a training table.
/// Equivalent to Python's data fetch (lines 425-444).
async fn fetch_pre_train_data(
    pool: &PgPool,
    table_name: &str,
    limit: i64,
    periods: &[usize],
) -> Result<Vec<TrainingDataRow>, sqlx::Error> {
    let change_cols: Vec<String> = periods
        .iter()
        .map(|p| format!("\"change{}\"", p))
        .collect();
    let col_list = change_cols.join(", ");

    let sql = format!(
        "SELECT open_timestamp, \"change\", {} FROM ( \
         SELECT open_timestamp, \"change\", {} FROM {} \
         ORDER BY open_timestamp DESC LIMIT {} \
         ) AS sub ORDER BY open_timestamp ASC",
        col_list, col_list, table_name, limit
    );

    let rows = sqlx::query(&sql).fetch_all(pool).await?;
    let mut result = Vec::with_capacity(rows.len());
    for row in &rows {
        let open_ts: i64 = row.get("open_timestamp");
        let change: f64 = row.get("change");
        let mut changes = Vec::with_capacity(periods.len());
        for p in periods {
            let col_name = format!("change{}", p);
            let val: f64 = row.get(&*col_name);
            changes.push(val);
        }
        result.push(TrainingDataRow { open_timestamp: open_ts, change, changes });
    }
    Ok(result)
}

/// Find the index of a period value in the periods list.
fn find_period_index(periods: &[usize], target: usize) -> Option<usize> {
    periods.iter().position(|&p| p == target)
}

/// Pre-processed training data with standardized features and computed scaler.
struct PreparedTrainingData {
    features: Vec<Vec<f64>>,
    targets: Vec<f64>,
    scaler_mean: Vec<f64>,
    scaler_scale: Vec<f64>,
}

/// Prepare training data: bin targets, compute Y_change15, standardize features.
///
/// `num_features`: number of change columns to use (336 for model 1, 96 for model 2).
/// `target_type`: "change_int" (model 1) or "change15_int" (model 2).
/// `shifts`: rows ahead for Y_change15 (180 = 12*15).
fn prepare_training_data(
    data: &[TrainingDataRow],
    periods: &[usize],
    num_features: usize,
    target_type: &str,
    shifts: usize,
) -> PreparedTrainingData {
    let n = data.len();
    if n == 0 {
        return PreparedTrainingData {
            features: vec![],
            targets: vec![],
            scaler_mean: vec![],
            scaler_scale: vec![],
        };
    }

    // Build features matrix (raw, not yet standardized)
    let mut features_raw: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in data {
        let feats: Vec<f64> = row.changes.iter().take(num_features).copied().collect();
        features_raw.push(feats);
    }

    // Compute targets for ALL rows (need full range for shift operations)
    let targets_full: Vec<f64> = match target_type {
        "change_int" => data.iter().map(|row| feature_method::bin_change(row.change)).collect(),
        "change15_int" => {
            let c180_idx = find_period_index(periods, 180).unwrap_or(0);
            let mut tgt = vec![0.0; n];
            for i in 0..n {
                if i + shifts < n {
                    let future = data[i + shifts].changes[c180_idx];
                    tgt[i] = feature_method::bin_change15(future * 100.0);
                }
            }
            tgt
        }
        _ => vec![0.0; n],
    };

    // Trim trailing rows until last open_timestamp is 2h-aligned
    let mut valid_end = n;
    while valid_end > 0 && data[valid_end - 1].open_timestamp % 7200 != 0 {
        valid_end -= 1;
    }
    let effective_end = valid_end;

    // Take last 720 rows (matching Python model_length = [720])
    let take_rows = 720;
    let start_idx = if effective_end > take_rows {
        effective_end - take_rows
    } else {
        0
    };

    // Subset features and targets
    let mut subset_features: Vec<Vec<f64>> = Vec::new();
    let mut subset_targets: Vec<f64> = Vec::new();

    // For change15_int, re-check shift validity on the subset
    for i in start_idx..effective_end {
        let tgt = match target_type {
            "change15_int" => {
                if i + shifts < n {
                    // Recompute using original data for shift
                    let c180_idx = find_period_index(periods, 180).unwrap_or(0);
                    let future = data[i + shifts].changes[c180_idx];
                    feature_method::bin_change15(future * 100.0)
                } else {
                    continue; // skip rows without enough future data
                }
            }
            _ => targets_full[i],
        };
        subset_features.push(features_raw[i].clone());
        subset_targets.push(tgt);
    }

    let n_rows = subset_features.len();
    let n_feats = num_features;

    // Compute scaler (mean, std) from subset
    let mut scaler_mean = vec![0.0; n_feats];
    let mut scaler_scale = vec![0.0; n_feats];
    if n_rows > 0 {
        for row in &subset_features {
            for (j, &val) in row.iter().enumerate() {
                scaler_mean[j] += val;
            }
        }
        for j in 0..n_feats {
            scaler_mean[j] /= n_rows as f64;
        }
        for row in &subset_features {
            for (j, &val) in row.iter().enumerate() {
                let d = val - scaler_mean[j];
                scaler_scale[j] += d * d;
            }
        }
        for j in 0..n_feats {
            scaler_scale[j] = (scaler_scale[j] / n_rows as f64).sqrt();
            if scaler_scale[j] == 0.0 {
                scaler_scale[j] = 1.0;
            }
        }
    }

    // Standardize features
    let mut features_std: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for row in &subset_features {
        let std_row: Vec<f64> = row
            .iter()
            .enumerate()
            .map(|(j, &val)| (val - scaler_mean[j]) / scaler_scale[j])
            .collect();
        features_std.push(std_row);
    }

    PreparedTrainingData {
        features: features_std,
        targets: subset_targets,
        scaler_mean,
        scaler_scale,
    }
}

// ──────────────────── LSTM Training ────────────────────

/// LSTM model: LSTM(input→hidden, N layers) + Linear(hidden→1).
/// Matches Python's TimeSeriesLSTM class.
#[derive(Debug)]
struct TimeSeriesLSTM {
    lstm: nn::LSTM,
    fc: nn::Linear,
}

impl TimeSeriesLSTM {
    fn new(vs: &nn::Path, input_size: i64, hidden_size: i64, num_layers: i64) -> Self {
        let lstm = nn::lstm(
            vs / "lstm",
            input_size,
            hidden_size,
            nn::RNNConfig {
                num_layers,
                ..Default::default()
            },
        );
        let fc = nn::linear(
            vs / "fc",
            hidden_size,
            1,
            Default::default(),
        );
        TimeSeriesLSTM { lstm, fc }
    }
}

impl nn::Module for TimeSeriesLSTM {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // xs: (batch, seq_len, input_size)
        let (output, _state) = self.lstm.seq(xs);
        // output: (batch, seq_len, hidden_size) — take last timestep
        let seq_len = output.size()[1];
        let last = output.select(1, seq_len - 1);
        // Linear: (batch, hidden_size) → (batch, 1)
        last.apply(&self.fc)
    }
}

/// Train an LSTM model on the prepared data.
///
/// Equivalent to Python's `FinancialTransformerTrainer.train_multiple_epochs()`.
fn train_lstm_model(
    prepared: &PreparedTrainingData,
    input_size: i64,
    hidden_size: i64,
    num_layers: i64,
    seq_len: i64,
    num_epochs: i64,
    learning_rate: f64,
    device: Device,
) -> Result<nn::VarStore, Box<dyn std::error::Error>> {
    let n_samples = prepared.features.len() as i64;
    if n_samples < seq_len {
        return Err(format!(
            "Not enough samples for training: {} < seq_len={}",
            n_samples, seq_len
        )
        .into());
    }

    let n_seq = n_samples - seq_len + 1;
    let n_features = input_size;
    let batch_size: i64 = 32;

    let vs = nn::VarStore::new(device);
    let model = TimeSeriesLSTM::new(&vs.root(), input_size, hidden_size, num_layers);
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0_f64;
        let mut n_batches = 0_i64;

        let mut start: i64 = 0;
        while start < n_seq {
            let end = std::cmp::min(start + batch_size, n_seq);
            let batch_actual = end - start;
            if batch_actual <= 0 {
                break;
            }

            // Build batch feature tensor: (batch_actual, seq_len, n_features)
            let cap = (batch_actual * seq_len * n_features) as usize;
            let mut x_data: Vec<f32> = Vec::with_capacity(cap);
            let mut y_data: Vec<f32> = Vec::with_capacity(batch_actual as usize);

            for b in start..end {
                for t in 0..seq_len {
                    let idx = (b + t) as usize;
                    for f in 0..n_features as usize {
                        x_data.push(prepared.features[idx][f] as f32);
                    }
                }
                let target_idx = (b + seq_len - 1) as usize;
                y_data.push(prepared.targets[target_idx] as f32);
            }

            let batch_x = Tensor::from_slice(&x_data)
                .to_device(device)
                .view([batch_actual, seq_len, n_features]);
            let batch_y = Tensor::from_slice(&y_data)
                .to_device(device)
                .view([batch_actual, 1]);

            let predictions = model.forward(&batch_x);
            let loss = predictions.mse_loss(&batch_y, tch::Reduction::Mean);
            opt.backward_step(&loss);

            let loss_val: f64 = f64::try_from(&loss).unwrap_or(0.0);
            epoch_loss += loss_val;
            n_batches += 1;
            start = end;
        }

        let avg_loss = epoch_loss / n_batches as f64;
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            eprintln!(
                "  Epoch {}/{}: Loss = {:.6}",
                epoch + 1,
                num_epochs,
                avg_loss
            );
        }
    }

    Ok(vs)
}

// ──────────────────── Model Export ────────────────────

/// Save a scaler vector to CSV (one value per line).
fn save_scaler_csv(path: &str, data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    for &val in data {
        writeln!(file, "{:.15e}", val)?;
    }
    Ok(())
}

/// Export a trained VarStore model to TorchScript via tracing.
fn export_model(
    vs: &mut nn::VarStore,
    input_size: i64,
    hidden_size: i64,
    num_layers: i64,
    model_path: &str,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Use seq_len=1 to match inference (Python predict uses seq_len=1).
    let dummy = Tensor::ones(&[1, 1, input_size], (Kind::Float, device));

    let model = TimeSeriesLSTM::new(&vs.root(), input_size, hidden_size, num_layers);

    // Freeze VarStore so parameters don't require grad during tracing
    vs.freeze();

    let mut forward_fn = |inputs: &[Tensor]| -> Vec<Tensor> {
        vec![model.forward(&inputs[0])]
    };

    let traced = CModule::create_by_tracing(
        "TimeSeriesLSTM",
        "forward",
        &[dummy],
        &mut forward_fn,
    )?;
    traced.save(model_path)?;
    eprintln!("  Model saved to {}", model_path);
    Ok(())
}

// ──────────────────── Main Loop ────────────────────

async fn update_loop() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = CONFIG.get().unwrap();
    let pool = init_db().await?;
    let (periods, target) = feature_method::compute_periods();
    let shifts_count = target;

    // ── Load initial raw data ──
    let query_short = format!(
        r#"
        SELECT "timestamp" AS snapshottime,
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {} ORDER BY "timestamp" DESC LIMIT 100
        ) AS sub ORDER BY "timestamp" ASC;
        "#,
        cfg.raw_table
    );

    let query_long = format!(
        r#"
        SELECT "timestamp" AS snapshottime,
               extract(epoch from "timestamp")::bigint AS timestamp,
               open, high, low, close, volume
        FROM (
            SELECT * FROM {} ORDER BY "timestamp" DESC LIMIT 30000
        ) AS sub ORDER BY "timestamp" ASC;
        "#,
        cfg.raw_table
    );

    let mut df4_temp = load_data(&pool, &query_short).await?;
    if !df4_temp.is_empty() {
        // Print diff counts (like Python's value_counts of timestamp diffs)
        eprintln!("Temp diff counts:");
        let mut diffs: Vec<i64> = df4_temp
            .windows(2)
            .map(|w| w[1].timestamp - w[0].timestamp)
            .collect();
        diffs.sort_unstable();
        let mut counts: Vec<(i64, usize)> = Vec::new();
        for &d in &diffs {
            if let Some(last) = counts.last_mut() {
                if last.0 == d {
                    last.1 += 1;
                    continue;
                }
            }
            counts.push((d, 1));
        }
        for (d, c) in &counts {
            eprintln!("  {}: {}", d, c);
        }
    } else {
        eprintln!("No historical data found. Will start fresh from Binance.");
        let now_ts = Utc::now().timestamp();
        df4_temp.push(Record5s {
            snapshot_time: utc_to_hkt_naive(DateTime::from_timestamp(now_ts - 5, 0).unwrap()),
            timestamp: now_ts - 5,
            open: 0.0,
            high: 0.0,
            low: 0.0,
            close: 0.0,
            volume: 0.0,
        });
    }

    let mut df4_old = load_data(&pool, &query_long).await?;
    if !df4_old.is_empty() {
        eprintln!("Old diff counts:");
        let mut diffs: Vec<i64> = df4_old
            .windows(2)
            .map(|w| w[1].timestamp - w[0].timestamp)
            .collect();
        diffs.sort_unstable();
        let mut counts: Vec<(i64, usize)> = Vec::new();
        for &d in &diffs {
            if let Some(last) = counts.last_mut() {
                if last.0 == d {
                    last.1 += 1;
                    continue;
                }
            }
            counts.push((d, 1));
        }
        for (d, c) in &counts {
            eprintln!("  {}: {}", d, c);
        }
    }

    // ── Load last training positions ──
    let (mut last_buy_ts, _) = get_last_open(&pool, &cfg.buy_table).await?;
    let (mut last_sell_ts, _) = get_last_open(&pool, &cfg.sell_table).await?;

    // Spread reference: timestamp of the last pre_train row (like Python's model['open_timestamp'].values[-1]).
    // Updated only during pre_train so spread grows as time passes.
    let mut spread_ref_ts: i64 = if last_buy_ts > 0 { last_buy_ts } else { 0 };

    // ── Prepare model/scaler containers for lazy loading ──
    let mut model1_opt: Option<LSTMModel> = None;
    let mut model2_opt: Option<LSTMModel> = None;
    let mut scaler1_mean: Vec<f64> = Vec::new();
    let mut scaler1_scale: Vec<f64> = Vec::new();
    let mut scaler2_mean: Vec<f64> = Vec::new();
    let mut scaler2_scale: Vec<f64> = Vec::new();
    let mut models_loaded = false;

    // Probe CUDA: dlopen libtorch_cuda.so so the libtorch runtime sees CUDA
    let device = {
        let cuda_handle = unsafe {
            libc::dlopen(b"libtorch_cuda.so\0".as_ptr() as *const libc::c_char,
                         libc::RTLD_NOW | libc::RTLD_GLOBAL)
        };
        if cuda_handle.is_null() {
            let err = unsafe { std::ffi::CStr::from_ptr(libc::dlerror()) };
            eprintln!("CUDA library not loadable: {:?}", err);
            Device::Cpu
        } else {
            eprintln!("CUDA library loaded via dlopen, probing...");
            let _ = cuda_handle;
            Device::cuda_if_available()
        }
    };
    match device {
        Device::Cuda(id) => eprintln!("CUDA available, using device {}", id),
        _ => eprintln!("CUDA not available, using CPU"),
    }

    let column_b_periods = feature_method::compute_column_b_periods();
    eprintln!("columnB periods: {}", column_b_periods.len());
    eprintln!("spread thresholds: inference={}, retrain={}", cfg.spread_target, cfg.spread_retrain);

    // ── Binance HTTP client ──
    let client = Client::new();
    let sem = Arc::new(Semaphore::new(cfg.semaphore_limit));

    // ── Main loop ──
    loop {
        let loop_start = Instant::now();
        let wall_start = Utc::now();
        let wall_start_f = wall_start.timestamp() as f64
            + wall_start.timestamp_subsec_millis() as f64 / 1000.0;

        // Determine how much data we need to fetch
        let last_ts = df4_temp.last().map(|r| r.timestamp).unwrap_or(0);
        let now_ts = Utc::now().timestamp();
        let total_seconds = std::cmp::max(0, now_ts - last_ts);
        let total_loops = total_seconds / 1000 + 1;

        let mut df_insert: Vec<Record5s> = Vec::new();

        for i in 0..total_loops {
            let start_ms = (last_ts + i * 1000) * 1000;

            let raw = {
                let _permit = sem.acquire().await.unwrap();
                fetch_klines(&client, &cfg.symbol, "1s", start_ms, 1000).await?
            };

            if raw.is_empty() {
                continue;
            }

            let mut df_5s = resample_to_n_seconds(&raw, 5);
            // Deduplicate by timestamp
            df_5s.dedup_by_key(|r| r.timestamp);
            df_insert.append(&mut df_5s);
        }

        // Insert 5s data to DB
        if !df_insert.is_empty() {
            insert_5s_batch(&pool, &df_insert).await?;
        }

        // Update in-memory state
        if !df_insert.is_empty() {
            df4_temp = df_insert.clone();
            df4_old.append(&mut df_insert);
            df4_old.sort_unstable_by_key(|r| r.timestamp);
            df4_old.dedup_by_key(|r| r.timestamp);
            if df4_old.len() > 25000 {
                df4_old = df4_old.split_off(df4_old.len() - 25000);
            }
        }

        if df4_old.is_empty() {
            let elapsed = loop_start.elapsed().as_secs_f64();
            let sleep_sec = 5.5 - (wall_start_f % 5.0);
            eprintln!("cost: {:.1}", elapsed);
            if elapsed < sleep_sec {
                tokio::time::sleep(Duration::from_secs_f64(sleep_sec - elapsed)).await;
            }
            continue;
        }

        // ── Feature engineering ──
        let features = feature_method::compute_features(&df4_old, &periods);
        let mut spread = features.last().map(|f| {
            let ref_ts = if spread_ref_ts > 0 { spread_ref_ts } else { features[0].timestamp };
            f.timestamp - ref_ts
        }).unwrap_or(0);
        if features.len() < shifts_count {
            let elapsed = loop_start.elapsed().as_secs_f64();
            let sleep_sec = 5.5 - (wall_start_f % 5.0);
            eprintln!("cost: {:.1}", elapsed);
            if elapsed < sleep_sec {
                tokio::time::sleep(Duration::from_secs_f64(sleep_sec - elapsed)).await;
            }
            continue;
        }

        // ── Buy training data ──
        let buy_rows = feature_method::prepare_train_data(&features, &periods, target, shifts_count, "buy", false);
        let buy_new: Vec<_> = buy_rows
            .into_iter()
            .filter(|r| r.open_timestamp > last_buy_ts)
            .collect();

        if !buy_new.is_empty() {
            let to_insert: Vec<TrainingRow> = buy_new; // consume
            insert_train_batch(&pool, &to_insert, &periods, &cfg.buy_table).await?;
            let (ts, _) = get_last_open(&pool, &cfg.buy_table).await?;
            last_buy_ts = ts;
            if ts > 0 {
                let d = utc_to_hkt_naive(DateTime::from_timestamp(ts, 0).unwrap());
                eprintln!("buy_last_open: {}", d);
            }
        }

        // ── Sell training data ──
        let sell_rows =
            feature_method::prepare_train_data(&features, &periods, target, shifts_count, "sell", true);
        let sell_new: Vec<_> = sell_rows
            .into_iter()
            .filter(|r| r.open_timestamp > last_sell_ts)
            .collect();

        if !sell_new.is_empty() {
            let to_insert: Vec<TrainingRow> = sell_new;
            insert_train_batch(&pool, &to_insert, &periods, &cfg.sell_table).await?;
            let (ts, _) = get_last_open(&pool, &cfg.sell_table).await?;
            last_sell_ts = ts;
            if ts > 0 {
                let d = utc_to_hkt_naive(DateTime::from_timestamp(ts, 0).unwrap());
                eprintln!("sell_last_open: {}", d);
            }
        }

        // ── Pre-train: train models on startup (one-time) ──
        if cfg.pre_train && !PRE_TRAIN_DONE.load(Ordering::Relaxed) {
            eprintln!("=== PRE-TRAIN START ===");

            // 1. Trim tables and update last timestamps
            // Keep max() of pre/post-trim ts so insert filter doesn't go backward
            eprintln!("pre-train: trimming to 2h alignment");
            let pre_trim_buy_ts = last_buy_ts;
            trim_to_aligned(&pool, &cfg.buy_table).await;
            trim_to_aligned(&pool, &cfg.sell_table).await;
            let (ts, _) = get_last_open(&pool, &cfg.buy_table).await?;
            last_buy_ts = std::cmp::max(ts, pre_trim_buy_ts);
            let pre_trim_sell_ts = last_sell_ts;
            let (ts, _) = get_last_open(&pool, &cfg.sell_table).await?;
            last_sell_ts = std::cmp::max(ts, pre_trim_sell_ts);

            // 2. Fetch training data from buy table
            let train_data = fetch_pre_train_data(
                &pool, &cfg.buy_table, 10000, &periods
            ).await?;
            eprintln!("pre-train: fetched {} rows", train_data.len());

            // Compute spread_ref_ts: last pre_train row's timestamp after 2h alignment
            // (matches Python: model['open_timestamp'].values[-1])
            {
                let mut ref_end = train_data.len();
                while ref_end > 0 && train_data[ref_end - 1].open_timestamp % 7200 != 0 {
                    ref_end -= 1;
                }
                if ref_end > 0 {
                    spread_ref_ts = train_data[ref_end - 1].open_timestamp;
                    eprintln!("pre-train: spread_ref_ts = {}", spread_ref_ts);
                }
            }

            if train_data.len() >= 1000 {
                let shifts: usize = 12 * 15; // 180 rows = 15 min
                let col_b = feature_method::compute_column_b_periods();
                let input_size_1 = periods.len() as i64;   // 336
                let input_size_2 = col_b.len() as i64;      // 96
                let hidden_size = input_size_1;
                let seq_len: i64 = 12;
                let num_epochs: i64 = 100;
                let lr: f64 = 0.001;

                // 3. Model 1 (change_int)
                eprintln!("pre-train: model 1 (LSTM {}->{}, 4 layers)...",
                    input_size_1, hidden_size);
                let prep1 = prepare_training_data(
                    &train_data, &periods, input_size_1 as usize, "change_int", shifts
                );
                let mut vs1 = train_lstm_model(
                    &prep1, input_size_1, hidden_size, 4, seq_len, num_epochs, lr, device,
                )?;
                export_model(&mut vs1, input_size_1, hidden_size, 4, &cfg.model1_path, device)?;
                save_scaler_csv(&cfg.scaler1_mean_path, &prep1.scaler_mean)?;
                save_scaler_csv(&cfg.scaler1_scale_path, &prep1.scaler_scale)?;

                // 4. Model 2 (change15_int)
                eprintln!("pre-train: model 2 (LSTM {}->{}, 4 layers)...",
                    input_size_2, input_size_2);
                let prep2 = prepare_training_data(
                    &train_data, &col_b, input_size_2 as usize, "change15_int", shifts
                );
                let mut vs2 = train_lstm_model(
                    &prep2, input_size_2, input_size_2, 4, seq_len, num_epochs, lr, device,
                )?;
                export_model(&mut vs2, input_size_2, input_size_2, 4, &cfg.model2_path, device)?;
                save_scaler_csv(&cfg.scaler2_mean_path, &prep2.scaler_mean)?;
                save_scaler_csv(&cfg.scaler2_scale_path, &prep2.scaler_scale)?;

                eprintln!("=== PRE-TRAIN COMPLETE ===");

                // Recompute spread with the updated spread_ref_ts so this loop iteration
                // uses the correct spread for inference.
                spread = features.last().map(|f| {
                    let ref_ts = if spread_ref_ts > 0 { spread_ref_ts } else { features[0].timestamp };
                    f.timestamp - ref_ts
                }).unwrap_or(0);
            } else {
                eprintln!("pre-train: only {} rows (need >= 1000), skipping", train_data.len());
            }
            PRE_TRAIN_DONE.store(true, Ordering::Relaxed);
        }

        // ── Lazy model loading (once, after pre_train if enabled) ──
        if !models_loaded {
            eprintln!("Loading ML models and scalers...");
            model1_opt = Some(LSTMModel::load(&cfg.model1_path, device)?);
            model2_opt = Some(LSTMModel::load(&cfg.model2_path, device)?);
            scaler1_mean = load_csv_f64(&cfg.scaler1_mean_path)?;
            scaler1_scale = load_csv_f64(&cfg.scaler1_scale_path)?;
            scaler2_mean = load_csv_f64(&cfg.scaler2_mean_path)?;
            scaler2_scale = load_csv_f64(&cfg.scaler2_scale_path)?;
            eprintln!(
                "Models loaded: model1={} features, model2={} features",
                scaler1_mean.len(),
                scaler2_mean.len()
            );
            models_loaded = true;
        }

        // ── Shadow references for inference code ──
        let model1 = model1_opt.as_ref().unwrap();
        let model2 = model2_opt.as_ref().unwrap();

        // ── Signal generation ──
        if let Some(last_feat) = features.last() {
            let latest_ts = last_feat.timestamp;

            if spread >= cfg.spread_retrain {
                // Retrain trigger: skip inference, write flag for external Python retrain process
                eprintln!("retrain: spread={} >= spread_retrain={}, writing .retrain_needed", spread, cfg.spread_retrain);
                let _ = std::fs::write(".retrain_needed", format!("{}", latest_ts));
            } else if spread >= cfg.spread_target {
                // Model1 inference (336 features)
                let std_features1 = standardize(&last_feat.changes, &scaler1_mean, &scaler1_scale);
                let pred1 = match model1.predict(&std_features1, 1) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("model1 predict error: {}", e);
                        f64::NAN
                    }
                };

                // Model2 inference (first 96 features = columnB subset)
                let col_b_len = column_b_periods.len();
                let col_b_features: Vec<f64> = last_feat.changes.iter()
                    .take(col_b_len).copied().collect();
                let std_features2 = standardize(&col_b_features, &scaler2_mean, &scaler2_scale);
                let pred2 = match model2.predict(&std_features2, 1) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("model2 predict error: {}", e);
                        f64::NAN
                    }
                };

                eprintln!("signal: {:.4} {:.4} ts={} spread={}", pred1, pred2, latest_ts, spread);

                // Buy signal — always insert (actual or zeros, matching Python behavior)
                if pred1 >= cfg.buy_threshold && pred2 >= cfg.buy_threshold2 {
                    insert_signal(
                        &pool, &cfg.buy_signal_table, &cfg.symbol,
                        last_feat.close, pred1, pred2, last_feat.timestamp, 0.0,
                    ).await?;
                    eprintln!(">>> BUY signal inserted: {:.4} {:.4}", pred1, pred2);
                } else {
                    insert_signal(
                        &pool, &cfg.buy_signal_table, &cfg.symbol,
                        last_feat.close, 0.0, 0.0, last_feat.timestamp, 0.0,
                    ).await?;
                    eprintln!("no buy signal — zeros inserted: {:.4} {:.4}", pred1, pred2);
                }

                // Sell signal — always insert (actual or zeros)
                if pred1 <= -cfg.sell_threshold && pred2 <= -cfg.sell_threshold2 {
                    insert_signal(
                        &pool, &cfg.sell_signal_table, &cfg.symbol,
                        last_feat.close, pred1, pred2, last_feat.timestamp, 0.0,
                    ).await?;
                    eprintln!(">>> SELL signal inserted: {:.4} {:.4}", pred1, pred2);
                } else {
                    insert_signal(
                        &pool, &cfg.sell_signal_table, &cfg.symbol,
                        last_feat.close, 0.0, 0.0, last_feat.timestamp, 0.0,
                    ).await?;
                    eprintln!("no sell signal — zeros inserted: {:.4} {:.4}", pred1, pred2);
                }
            } else {
                eprintln!("no signal (spread={} < {})", spread, cfg.spread_target);
            }
        }

        // ── Sleep until next 5s boundary ──
        let elapsed = loop_start.elapsed().as_secs_f64();
        let sleep_sec = 5.5 - (Utc::now().timestamp() as f64 % 5.0);
        eprintln!("cost: {:.1} spread={}", elapsed, spread);
        if elapsed < sleep_sec {
            tokio::time::sleep(Duration::from_secs_f64(sleep_sec - elapsed)).await;
        }
    }
}

// ──────────────────── Entry Point ────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // env.txt lives at the repo root (outside Rust/). Prefer an explicit
    // ENV_FILE path, then ./env.txt (run from repo root), then ../env.txt
    // (run from inside Rust/).
    let env_path = std::env::var("ENV_FILE").unwrap_or_default();
    let env_path = if !env_path.is_empty() {
        env_path
    } else if std::path::Path::new("env.txt").is_file() {
        "env.txt".to_string()
    } else {
        "../env.txt".to_string()
    };
    CONFIG.set(Config::load(&env_path)?).map_err(|_| "config already initialized")?;
    update_loop().await
}
