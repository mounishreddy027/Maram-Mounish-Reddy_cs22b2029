-- Enable the TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create table for raw tick data
CREATE TABLE IF NOT EXISTS ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol    VARCHAR(20) NOT NULL,
    price     DECIMAL NOT NULL,
    size      DECIMAL NOT NULL
);

-- Create a TimescaleDB hypertable (partitioned by time)
SELECT create_hypertable('ticks', 'timestamp', if_not_exists => TRUE);

-- Add index for faster queries by symbol and time
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_timestamp ON ticks (symbol, timestamp DESC);

-- Create a materialized view for 1-minute OHLC bars
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlc_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS timestamp,
    symbol,
    FIRST(price, timestamp) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, timestamp) AS close,
    SUM(size) AS volume
FROM ticks
GROUP BY 1, 2
WITH NO DATA;

-- Create a materialized view for 5-minute OHLC bars
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlc_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', timestamp) AS timestamp,
    symbol,
    FIRST(price, timestamp) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, timestamp) AS close,
    SUM(size) AS volume
FROM ticks
GROUP BY 1, 2
WITH NO DATA;

-- Create policies to automatically refresh the views
SELECT add_continuous_aggregate_policy('ohlc_1m',
    start_offset => INTERVAL '15 minute',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('ohlc_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minute',
    schedule_interval => INTERVAL '5 minute');