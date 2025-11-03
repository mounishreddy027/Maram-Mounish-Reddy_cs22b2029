import os
import psycopg2
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from fastapi import FastAPI, Query, HTTPException
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import numpy as np
from pykalman import KalmanFilter
from sklearn.linear_model import HuberRegressor, TheilSenRegressor # <-- NEW IMPORT

# --- App and DB Setup ---
app = FastAPI()
DATABASE_URL = os.environ.get("DATABASE_URL")
db_pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)

@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# --- Helper Functions ---
def fetch_data_as_df(query, params=None):
    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params=params)

def get_pair_prices(sym1, sym2, timeframe):
    table_map = {"1m": "ohlc_1m", "5m": "ohlc_5m"}
    if timeframe not in table_map:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    table = table_map[timeframe]
    query = f"""
        SELECT timestamp, symbol, close
        FROM {table}
        WHERE symbol IN (%s, %s)
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    df = fetch_data_as_df(query, (sym1, sym2))
    
    if df.empty:
        return pd.DataFrame()

    df_pivot = df.pivot(index='timestamp', columns='symbol', values='close').sort_index()
    
    if sym1 not in df_pivot.columns or sym2 not in df_pivot.columns:
        return pd.DataFrame()
        
    df_pivot = df_pivot[[sym1, sym2]].ffill().dropna()
    return df_pivot

# --- API Endpoints ---

@app.get("/symbols")
def get_symbols():
    query = "SELECT DISTINCT symbol FROM ticks;"
    df = fetch_data_as_df(query)
    return {"symbols": df['symbol'].tolist()}

@app.get("/ohlc")
def get_ohlc(
    symbol: str,
    timeframe: str = Query("1s", enum=["1s", "1m", "5m"])
):
    if timeframe == "1s":
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
        table = "ohlc_1m" if timeframe == "1m" else "ohlc_5m"
        query = f"""
            SELECT * FROM {table}
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 500;
        """
        params = (symbol,)
        
    df = fetch_data_as_df(query, params)
    df = df.sort_values(by='timestamp', ascending=True) # Sort ascending for charts
    return df.to_dict('records')


@app.get("/analytics/pair")
def get_pair_analytics(
    sym1: str,
    sym2: str,
    timeframe: str = Query("1m", enum=["1m", "5m"]),
    window: int = Query(20, gt=0),
    # --- UPDATED PARAMETER ---
    regression_type: str = Query("ols", enum=["ols", "kalman", "huber", "theilsen"])
):
    """Computes OLS, Kalman, Huber, or Theil-Sen hedge ratio, spread, z-score, and correlation."""
    df = get_pair_prices(sym1, sym2, timeframe)
    
    if df.empty or len(df) < window:
        raise HTTPException(status_code=404, detail="Not enough data for analytics")

    hedge_ratio = 0
    y = df[sym1].values
    X = df[sym2].values.reshape(-1, 1) # sklearn needs 2D array
    
    # --- UPDATED REGRESSION LOGIC ---
    
    if regression_type == "ols":
        X_with_const = sm.add_constant(df[sym2]) # OLS needs constant
        model = sm.OLS(y, X_with_const).fit()
        hedge_ratio = model.params.get(sym2, 0)
        intercept = model.params.get('const', 0)
        df['spread'] = df[sym1] - (hedge_ratio * df[sym2] + intercept)

    elif regression_type == "huber":
        model = HuberRegressor().fit(X, y)
        hedge_ratio = model.coef_[0]
        intercept = model.intercept_
        df['spread'] = df[sym1] - (hedge_ratio * df[sym2] + intercept)

    elif regression_type == "theilsen":
        model = TheilSenRegressor().fit(X, y)
        hedge_ratio = model.coef_[0]
        intercept = model.intercept_
        df['spread'] = df[sym1] - (hedge_ratio * df[sym2] + intercept)
    
    else: # regression_type == "kalman"
        obs_matrix = np.vstack([df[sym2], np.ones(len(df[sym2]))]).T[:, np.newaxis, :]

        kf = KalmanFilter(
            transition_matrices = [[1, 0], [0, 1]],
            observation_matrices = obs_matrix,
            initial_state_mean = [0, 0],
            initial_state_covariance = [[1, 0], [0, 1]],
            transition_covariance = [[1e-5, 0], [0, 1e-5]],
            observation_covariance = 1.0
        )
        
        state_means, _ = kf.filter(y)
        
        df['hedge_ratio'] = state_means[:, 0]
        df['intercept'] = state_means[:, 1]
        
        df['spread'] = df[sym1] - (df['hedge_ratio'] * df[sym2] + df['intercept'])
        hedge_ratio = state_means[-1, 0] # The latest hedge ratio

    # -------------------------------------

    # 2. Z-Score (based on the calculated spread)
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

    # Clean data for JSON response
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "hedge_ratio": float(hedge_ratio),
        "adf_stats": adf_stats,
        "data": df.reset_index().to_dict('records')
    }

