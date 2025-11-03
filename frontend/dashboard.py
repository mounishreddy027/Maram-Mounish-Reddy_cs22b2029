import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
from sklearn.linear_model import HuberRegressor, TheilSenRegressor # <-- NEW IMPORT

# --- Page Configuration ---
st.set_page_config(
    page_title="Quant Analytics Dashboard",
    layout="wide"
)

# --- API Communication ---
API_BASE_URL = "http://api:8000" # URL of our FastAPI backend

@st.cache_data(ttl=60)
def get_symbols():
    """Get list of available symbols from API."""
    try:
        res = requests.get(f"{API_BASE_URL}/symbols")
        res.raise_for_status()
        return res.json().get("symbols", ["BTCUSDT", "ETHUSDT"])
    except requests.RequestException:
        return ["BTCUSDT", "ETHUSDT"] # Fallback

@st.cache_data(ttl=10)
def get_ohlc_data(symbol, timeframe):
    """Get OHLC data for a single symbol."""
    params = {"symbol": symbol, "timeframe": timeframe}
    try:
        res = requests.get(f"{API_BASE_URL}/ohlc", params=params)
        res.raise_for_status()
        df = pd.DataFrame(res.json())
        
        if df.empty:
            return df 

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.set_index('timestamp')
        # Convert the index (already in UTC) to IST
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        # API now sorts ASC
        return df
        
    except requests.RequestException as e:
        st.error(f"Error fetching OHLC data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=10)
def get_pair_analytics(sym1, sym2, timeframe, window, regression_type):
    """Get pair analytics (z-score, OLS, etc.) from API."""
    params = {
        "sym1": sym1, 
        "sym2": sym2, 
        "timeframe": timeframe, 
        "window": window,
        "regression_type": regression_type
    }
    try:
        res = requests.get(f"{API_BASE_URL}/analytics/pair", params=params)
        
        if res.status_code == 404:
            return pd.DataFrame(), 0, {}
        elif res.status_code == 500:
            st.error("Whoops! The analytics server ran into an error. Please check the `api` service logs for details.")
            return pd.DataFrame(), 0, {}

        res.raise_for_status() 
        
        data = res.json()
        df = pd.DataFrame(data['data'])

        if df.empty:
            return pd.DataFrame(), 0, {}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.set_index('timestamp')
        # Convert the index (already in UTC) to IST
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        return df, data['hedge_ratio'], data['adf_stats']
        
    except requests.RequestException as e:
        st.error(f"Error connecting to the analytics API: {e}")
        return pd.DataFrame(), 0, {}

# --- Plotting Functions ---
def plot_price(df, symbol):
    """Plots a candlestick chart for a single symbol."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    
    fig.update_layout(
        title=f"{symbol} Price",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        uirevision="foo"
    )
    return fig

def plot_pair_analytics(df, sym1, sym2, regression_type):
    """Plots spread, z-score, correlation, and (optional) hedge ratio."""
    
    has_dynamic_hedge = 'hedge_ratio' in df.columns
    # Only show 4 rows if it's Kalman AND the hedge_ratio column exists
    rows = 4 if (regression_type == 'kalman' and has_dynamic_hedge) else 3
    titles = (f"{sym1}-{sym2} Spread", "Z-Score", "Rolling Correlation")
    if rows == 4:
        titles = ("Dynamic Hedge Ratio",) + titles

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.05
    )
    
    row_num = 1
    if rows == 4:
        fig.add_trace(go.Scatter(x=df.index, y=df['hedge_ratio'], name='Kalman Beta'), row=row_num, col=1)
        row_num += 1

    fig.add_trace(go.Scatter(x=df.index, y=df['spread'], name='Spread'), row=row_num, col=1)
    row_num += 1

    fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score'), row=row_num, col=1)
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=row_num, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="red", row=row_num, col=1)
    row_num += 1

    fig.add_trace(go.Scatter(x=df.index, y=df['correlation'], name='Correlation'), row=row_num, col=1)
    
    fig.update_layout(
        height=800 if rows == 4 else 600, 
        title_text=f"Pair Analytics ({regression_type.upper()})",
        uirevision="bar"
    )
    return fig

# --- UPDATED Local Analytics Function ---
def run_local_pair_analytics(df_pivot, sym1, sym2, window, regression_type):
    """Runs the same analytics as the API, but locally on a DataFrame."""
    df = df_pivot.copy()
    
    if df.empty or len(df) < window:
        st.warning(f"Not enough data for a rolling window of {window}. Please upload a larger file.")
        return pd.DataFrame(), 0, {}

    hedge_ratio = 0
    y = df[sym1].values
    X = df[sym2].values.reshape(-1, 1) # sklearn needs 2D
    
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
        hedge_ratio = state_means[-1, 0] # latest
    
    # 2. Z-Score
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

    # Clean data for plotting
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df, float(hedge_ratio), adf_stats


# --- Main Dashboard UI ---

st.title(" Quant Analysis Dashboard ")

data_source = st.sidebar.radio("Data Source", ["Live Feed", "File Upload"])
st.sidebar.divider()

# --- UPDATED REGRESSION FORMATTING ---
REGRESSION_FORMAT_MAP = {
    "ols": "OLS (Static)",
    "kalman": "Kalman (Dynamic)",
    "huber": "Huber (Robust)",
    "theilsen": "Theil-Sen (Robust)"
}
REGRESSION_OPTIONS = ["ols", "huber", "theilsen", "kalman"]

if data_source == "Live Feed":
    
    st.sidebar.header("Live Controls")
    analysis_type = st.sidebar.radio("Analysis Type", ["Single Symbol", "Pair Trading"])
    available_symbols = get_symbols()

    if analysis_type == "Single Symbol":
        st.sidebar.subheader("Symbol Selection")
        symbol = st.sidebar.selectbox("Symbol", available_symbols, index=0)
        timeframe = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"], index=1)
        
        st.header(f"Live Analytics for: {symbol}")
        
        ohlc_data = get_ohlc_data(symbol, timeframe)
        
        if not ohlc_data.empty:
            try:
                latest_price = ohlc_data['close'].iloc[-1]
                latest_volume = ohlc_data['volume'].iloc[-1]
                
                st.subheader("Live Stats")
                col1, col2 = st.columns(2)
                col1.metric("Latest Price", f"${latest_price:,.2f}")
                col2.metric(f"Latest Volume ({timeframe})", f"{latest_volume:,.0f}")
            except Exception:
                pass
            
            st.divider()

            st.plotly_chart(plot_price(ohlc_data, symbol), use_container_width=True)
            
            st.download_button(
                label="Download Data as CSV",
                data=ohlc_data.to_csv().encode('utf-8'),
                file_name=f"{symbol}_{timeframe}_ohlc.csv",
                mime='text/csv',
            )
            st.dataframe(ohlc_data.tail())
        else:
            st.warning("No data available for this selection.")

    else:
        # --- Pair Trading View ---
        st.sidebar.subheader("Symbol Selection")
        sym1 = st.sidebar.selectbox("Symbol 1 (Y)", available_symbols, index=0)
        sym2 = st.sidebar.selectbox("Symbol 2 (X)", available_symbols, index=1)
        
        st.sidebar.subheader("Analytics Parameters")
        # --- UPDATED REGRESSION TYPE SELECTOR ---
        regression_type = st.sidebar.radio(
            "Regression Type",
            REGRESSION_OPTIONS,
            format_func=lambda x: REGRESSION_FORMAT_MAP.get(x, x.upper())
        )
        timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m"], index=0)
        window = st.sidebar.number_input("Rolling Window", min_value=5, max_value=200, value=20) 
        
        st.header(f"Live Pair Analytics: {sym1} / {sym2}")
        
        pair_data, hedge_ratio, adf_stats = get_pair_analytics(sym1, sym2, timeframe, window, regression_type)
        
        if not pair_data.empty:
            try:
                latest_zscore = pair_data['z_score'].iloc[-1]
                latest_spread = pair_data['spread'].iloc[-1]
                
                st.subheader("Live Stats")
                col1, col2, col3 = st.columns(3)
                col1.metric("Latest Z-Score", f"{latest_zscore:.4f}")
                col2.metric("Latest Spread", f"{latest_spread:.4f}")
                
                if 'correlation' in pair_data.columns and not pd.isna(pair_data['correlation'].iloc[-1]):
                     latest_corr = pair_data['correlation'].iloc[-1]
                     col3.metric("Latest Correlation", f"{latest_corr:.4f}")

                st.subheader("Pair Configuration")
                col4, col5 = st.columns(2)
                col4.metric(f"Hedge Ratio ({REGRESSION_FORMAT_MAP[regression_type]})", f"{hedge_ratio:.4f}")
                col5.metric("Is Spread Stationary (ADF)?", "Yes" if adf_stats.get('is_stationary_5pct') else "No")

                if not pd.isna(latest_zscore) and abs(latest_zscore) > 2.0:
                    st.toast(f"🔔 ALERT: {sym1}/{sym2} Z-Score is {latest_zscore:.2f}!", icon="⚠️")

            except Exception as e:
                st.warning(f"Could not display live stats: {e}")

            st.divider()

            st.plotly_chart(plot_pair_analytics(pair_data, sym1, sym2, regression_type), use_container_width=True)

            st.download_button(
                label="Download Pair Data as CSV",
                data=pair_data.to_csv().encode('utf-8'),
                file_name=f"{sym1}_{sym2}_analytics.csv",
                mime='text/csv',
            )
            st.dataframe(pair_data.tail())
        else:
            st.warning("No data available for this selection. Please wait for data to be ingested.")

    time.sleep(5)
    st.rerun()

else: # data_source == "File Upload"
    
    st.sidebar.header("File Controls")
    uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Invalid CSV. Must contain columns: {', '.join(required_cols)}")
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                
                st.success("File uploaded and processed successfully!")
                
                analysis_type = st.radio("Analysis Type", ["Single Symbol", "Pair Trading"])
                
                if analysis_type == "Single Symbol":
                    st.sidebar.subheader("Symbol Selection")
                    symbol = st.sidebar.selectbox("Select Symbol", df['symbol'].unique())
                    
                    st.header(f"File Analytics for: {symbol}")
                    
                    ohlc_data = df[df['symbol'] == symbol]
                    
                    st.plotly_chart(plot_price(ohlc_data, symbol), use_container_width=True)
                    st.dataframe(ohlc_data.tail())

                else: # Pair Trading
                    st.sidebar.subheader("Symbol Selection")
                    symbols = df['symbol'].unique()
                    sym1 = st.sidebar.selectbox("Symbol 1 (Y)", symbols, index=0)
                    sym2 = st.sidebar.selectbox("Symbol 2 (X)", symbols, index=1 if len(symbols) > 1 else 0)
                    
                    st.sidebar.subheader("Analytics Parameters")
                    # --- UPDATED REGRESSION TYPE SELECTOR ---
                    regression_type = st.sidebar.radio(
                        "Regression Type",
                        REGRESSION_OPTIONS,
                        format_func=lambda x: REGRESSION_FORMAT_MAP.get(x, x.upper())
                    )
                    window = st.sidebar.number_input("Rolling Window", min_value=5, max_value=200, value=20) 
                    
                    st.header(f"File Pair Analytics: {sym1} / {sym2}")

                    df1 = df[df['symbol'] == sym1]['close']
                    df2 = df[df['symbol'] == sym2]['close']
                    pair_df = pd.concat([df1, df2], axis=1, join='inner')
                    pair_df.columns = [sym1, sym2]
                    
                    analytics_df, hedge_ratio, adf_stats = run_local_pair_analytics(pair_df, sym1, sym2, window, regression_type)
                    
                    if not analytics_df.empty:
                        try:
                            latest_zscore = analytics_df['z_score'].iloc[-1]
                            latest_spread = analytics_df['spread'].iloc[-1]
                            latest_corr = analytics_df['correlation'].iloc[-1]

                            st.subheader("Stats")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Latest Z-Score", f"{latest_zscore:.4f}")
                            col2.metric("Latest Spread", f"{latest_spread:.4f}")
                            col3.metric("Latest Correlation", f"{latest_corr:.4f}")

                            st.subheader("Pair Configuration")
                            col4, col5 = st.columns(2)
                            col4.metric(f"Hedge Ratio ({REGRESSION_FORMAT_MAP[regression_type]})", f"{hedge_ratio:.4f}")
                            col5.metric("Is Spread Stationary (ADF)?", "Yes" if adf_stats.get('is_stationary_5pct') else "No")

                        except Exception as e:
                            st.warning(f"Could not display stats: {e}")

                        st.divider()

                        st.plotly_chart(plot_pair_analytics(analytics_df, sym1, sym2, regression_type), use_container_width=True)
                        st.dataframe(analytics_df.tail())
                    else:
                        st.warning("Not enough data in the file to run pair analytics.")
                        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.stop()
    else:
        st.info("Upload a CSV file to begin analysis. The CSV must contain the columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`.")

