import os
import psycopg2
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from fastapi import FastAPI, Query, HTTPException
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import numpy as np

# --- App and DB Setup ---
app = FastAPI()
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create a connection pool
db_pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)

@contextmanager
def get_db_connection():
    """Get a connection from the pool."""
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# --- Helper Functions ---
def fetch_data_as_df(query, params=None):
    """Runs a query and returns a pandas DataFrame."""
    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params=params)

def get_pair_prices(sym1, sym2, timeframe):
    """Fetches and aligns price data for two symbols."""
    table_map = {"1m": "ohlc_1m", "5m": "ohlc_5m"}
    if timeframe not in table_map:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    table = table_map[timeframe]
    query = f"""
        SELECT timestamp, symbol, close
        FROM {table}
        WHERE symbol IN (%s, %s)
        ORDER BY timestamp DESC
        LIMIT 1000 -- Limit data for performance
    """
    df = fetch_data_as_df(query, (sym1, sym2))
    
    if df.empty:
        return pd.DataFrame()

    # Pivot to align timestamps
    df_pivot = df.pivot(index='timestamp', columns='symbol', values='close').sort_index()
    df_pivot = df_pivot.ffill().dropna() # Fill missing values and drop NaNs
    df_pivot.columns = [sym1, sym2]
    return df_pivot

# --- API Endpoints ---

@app.get("/symbols")
def get_symbols():
    """Get a list of all symbols with data."""
    query = "SELECT DISTINCT symbol FROM ticks;"
    df = fetch_data_as_df(query)
    return {"symbols": df['symbol'].tolist()}

@app.get("/ohlc")
def get_ohlc(
    symbol: str,
    timeframe: str = Query("1m", enum=["1s", "1m", "5m"])
):
    """Fetches OHLC data for a symbol and timeframe."""
    if timeframe == "1s":
        # 1s data is aggregated on-the-fly from raw ticks
        query = """
            SELECT
                time_bucket('1 second', timestamp) AS timestamp,
                FIRST(price, timestamp) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                LAST(price, timestamp) AS close,
                SUM(size) AS volume
            FROM ticks
            WHERE symbol = %s AND timestamp > (NOW() - INTERVAL '1 hour')
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 500;
        """
        params = (symbol,)
    else:
        # 1m and 5m data are read from pre-aggregated views
        table = "ohlc_1m" if timeframe == "1m" else "ohlc_5m"
        query = f"""
            SELECT * FROM {table}
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 500;
        """
        params = (symbol,)
        
    df = fetch_data_as_df(query, params)
    return df.to_dict('records')
@app.get("/analytics/pair")
def get_pair_analytics(
    sym1: str,
    sym2: str,
    timeframe: str = Query("1m", enum=["1m", "5m"]),
    window: int = Query(20, gt=0)
):
    """Computes OLS hedge ratio, spread, z-score, and correlation."""
    df = get_pair_prices(sym1, sym2, timeframe)

    if df.empty or len(df) < window:
        raise HTTPException(status_code=404, detail="Not enough data for analytics")

    # 1. OLS Hedge Ratio
    y = df[sym1]
    X = sm.add_constant(df[sym2]) # Add constant for intercept
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params.get(sym2, 0)

    # 2. Spread & Z-Score
    df['spread'] = df[sym1] - hedge_ratio * df[sym2]
    rolling_mean = df['spread'].rolling(window=window).mean()
    rolling_std = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - rolling_mean) / rolling_std

    # 3. Rolling Correlation
    df['correlation'] = df[sym1].rolling(window=window).corr(df[sym2])

    # 4. ADF Test on Spread
    try:
        adf_result = adfuller(df['spread'].dropna())
        adf_stats = {
            "test_statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "is_stationary_5pct": bool(adf_result[1] < 0.05)
        }
    except Exception:
        adf_stats = {"error": "ADF test failed to converge"}

    # --- THIS IS THE NEW FIX ---
    # 1. Replace all Infinity and -Infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # 2. Convert DataFrame to object type, then replace all NaN with None
    df = df.astype(object).where(pd.notna(df), None)
    # ---------------------------

    return {
        "hedge_ratio": float(hedge_ratio),
        "adf_stats": adf_stats,
        "data": df.reset_index().to_dict('records') 
    }