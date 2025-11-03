# 📊 Quant Analytics Dashboard

> A real-time quantitative trading analytics platform with live market data ingestion, statistical modeling, and interactive visualization



## 🎯 Overview

This project is a complete, end-to-end analytical application built for quantitative trading evaluation. It ingests real-time tick data from Binance, stores it efficiently, performs sophisticated quantitative analysis, and presents results through an interactive web dashboard.

### ✨ Key Highlights

- **Real-Time Data Pipeline** → Live WebSocket connection to Binance Futures
- **Time-Series Optimization** → PostgreSQL + TimescaleDB for high-performance storage
- **Automated Analytics** → Continuous aggregation and statistical modeling
- **Interactive Visualization** → Rich, explorable charts with Plotly
- **One-Command Deploy** → Complete stack orchestration with Docker Compose

---

## 🚀 Features

### 📡 Data Ingestion & Storage

- **Real-Time Stream**: WebSocket connection to Binance Futures (BTCUSDT & ETHUSDT)
- **Scalable Database**: PostgreSQL with TimescaleDB extension for optimized time-series data
- **Automated Sampling**: Continuous materialized views for 1-minute and 5-minute OHLC bars
- **Data Integrity**: Robust error handling and connection resilience

### 📈 Analytics & Modeling

#### Single-Symbol Analysis
- Candlestick charts at multiple timeframes (1s, 1m, 5m)
- Real-time price and volume tracking
- Historical trend visualization

#### Pair Trading Analytics
- **OLS Hedge Ratio**: Optimal portfolio weights for mean reversion strategies
- **Spread Calculation**: Price differential between correlated assets
- **Z-Score Analysis**: Statistical deviation detection for entry/exit signals
- **Rolling Correlation**: Dynamic relationship strength measurement
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test for mean reversion validation

### 🎛️ Interactive Dashboard

- **Dual Data Modes**:
  - 🔴 **Live Feed**: Real-time analysis from WebSocket stream
  - 📁 **File Upload**: Historical OHLCV CSV analysis
- **Live Metrics**: Dynamic stat boxes for price, volume, z-score, and spread
- **Smart Alerts**: On-screen notifications when z-score breaches ±2.0 threshold
- **Data Export**: Download processed data as CSV for further analysis

---

## 🏗️ Architecture

### Service-Oriented Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Streamlit)                    │
│                  Interactive Dashboard UI                    │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────┐
│                      API (FastAPI)                          │
│          Statistical Analysis & Data Serving                │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ PostgreSQL Protocol               │
         │                                   │
┌────────▼────────────┐           ┌─────────▼─────────────────┐
│    Aggregator       │           │  Database (PostgreSQL)    │
│  Refresh Manager    │◄──────────│  + TimescaleDB Extension  │
└─────────────────────┘           │  • Hypertables            │
                                  │  • Continuous Aggregates  │
                      ┌───────────┤  • Materialized Views     │
                      │           └───────────────────────────┘
                      │ SQL Insert
          ┌───────────▼──────────┐
          │      Ingestor        │
          │  WebSocket Client    │
          └───────────┬──────────┘
                      │ WebSocket
          ┌───────────▼──────────┐
          │   Binance Futures    │
          │    Market Stream     │
          └──────────────────────┘
```

### 🔧 Service Breakdown

| Service | Technology | Purpose |
|---------|-----------|---------|
| **db** | PostgreSQL + TimescaleDB | Time-series optimized storage with automatic aggregation |
| **ingestor** | Python + WebSockets | Real-time tick data collection from Binance |
| **aggregator** | Python | Continuous aggregate refresh orchestration |
| **api** | FastAPI | Statistical modeling and RESTful data serving |
| **frontend** | Streamlit + Plotly | Interactive visualization and user interface |

---

## 💡 Technology Stack

### Why These Technologies?

#### 🐳 **Docker Compose**
**Requirement**: Modular, scalable, loosely-coupled architecture  
**Advantage**: Five specialized microservices instead of a monolithic script. Services can be scaled, replaced, or extended independently. Fulfills "single-command execution" requirement.

#### 🗄️ **PostgreSQL + TimescaleDB**
**Requirement**: Efficient time-series data handling  
**Advantage**: 
- **Hypertables** optimize ingestion and querying of time-stamped data
- **Continuous Aggregates** perform 1m/5m sampling inside the database—far more efficient than Python loops
- Production-grade reliability and scalability

#### ⚡ **FastAPI**
**Requirement**: High-performance Python backend  
**Advantage**: 
- Asynchronous I/O for concurrent database operations
- 3-5x faster than Flask for I/O-bound workloads
- Automatic API documentation and data validation

#### 🎨 **Streamlit**
**Requirement**: Interactive data visualization  
**Advantage**: 
- Build complex UIs in ~200 lines vs. thousands in React
- Native integration with Plotly for interactive charts
- Built-in widgets (file upload, sidebar, radio buttons)

#### 📊 **Pandas + Statsmodels + NumPy**
**Requirement**: Quantitative analytics  
**Advantage**: 
- Industry-standard quantitative finance stack
- `statsmodels` provides exact required functions: `OLS()` and `adfuller()`
- Optimized numerical operations

#### 📉 **Plotly**
**Requirement**: Charts with zoom, pan, and hover  
**Advantage**: Purpose-built for interactive web visualizations. Matplotlib produces static images; Plotly provides rich interactivity out of the box.

---

## 🎮 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least 4GB of available RAM
- Ports 8501 (frontend) and 8000 (API) available

### 🚀 One-Command Launch

1. **Clone or download** this repository

2. **Navigate** to the project root directory
   ```bash
   cd quant-analytics-dashboard
   ```

3. **Start the stack**
   ```bash
   docker-compose up --build
   ```

4. **Access the dashb
