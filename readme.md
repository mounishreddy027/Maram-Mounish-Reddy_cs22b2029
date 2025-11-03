Quant Analytics Dashboard

This project is a complete, end-to-end analytical application built for the Quant Developer Evaluation. It ingests real-time tick data from Binance, stores it, performs quantitative analysis, and presents the results in an interactive web dashboard.

Features

Real-Time Ingestion: Connects to the Binance Futures WebSocket stream for live trade data (BTCUSDT & ETHUSDT).

Scalable Storage: Uses PostgreSQL with the TimescaleDB extension for high-performance time-series data storage.

Automated Sampling: Automatically aggregates ticks into 1-minute and 5-minute OHLC bars using TimescaleDB's continuous materialized views.

Backend API: A modular FastAPI backend serves all data and analytics via clean JSON endpoints.

Interactive Frontend: A Streamlit dashboard provides all visualizations and controls.

Dual Data Modes:

Live Feed: Analyzes real-time data from the WebSocket.

File Upload: Allows uploading and analyzing historical OHLCV CSV files.

Core Analytics:

Single-symbol candlestick charts (1s, 1m, 5m).

Pair trading analytics: OLS Hedge Ratio, Spread, Z-Score, and Rolling Correlation.

Stationarity testing with the Augmented Dickey-Fuller (ADF) test.

Live Stats & Alerting:

Real-time metric boxes for latest price, volume, z-score, and spread.

On-screen toast alert when the z-score breaches a threshold of ±2.0.

Data Export: "Download as CSV" buttons for all processed data.

Single-Command Execution: The entire 5-service application is orchestrated with docker-compose.

Architecture & Methodology

The application is built using a modular, multi-service architecture, orchestrated by docker-compose. This design separates concerns, ensuring the application is scalable and extensible.

The services are:

db (PostgreSQL + TimescaleDB): The core database. TimescaleDB was chosen for its powerful features for time-series data. Specifically, we use create_hypertable for efficient ticks storage and Continuous Aggregates (Materialized Views) to automatically and efficiently roll up data into 1-minute (ohlc_1m) and 5-minute (ohlc_5m) bars.

ingestor (Python): A standalone WebSocket client. Its only job is to connect to Binance, receive raw ticks, and insert them into the ticks table. It runs in its own container for resilience.

aggregator (Python): A simple service that runs in a loop and calls the refresh_continuous_aggregate procedure in the database. This ensures our 1m and 5m materialized views are kept up-to-date with the latest data.

api (Python/FastAPI): A high-performance backend API. It queries the database (both raw ticks and aggregated views), performs all statistical calculations (OLS, ADF, etc.), cleans the data (handles NaN/Infinity), and serves it to the frontend.

frontend (Python/Streamlit): The interactive user dashboard. It is a pure client of the api service and handles all plotting and user interaction. For the "File Upload" mode, it performs the same analytics as the API, but locally on the uploaded file.

(To generate this image, open the architecture.drawio file in a tool like draw.io (app.diagrams.net) and export it as architecture.png)

Technology Stack Justification

Here is the rationale for why each technology was chosen for this project:

1. Orchestration: docker-compose

Why: The assignment demands a modular, scalable, and loosely-coupled architecture.

Justification: Instead of building one giant, monolithic script, docker-compose allows us to run five separate, specialized services. This design is highly extensible (we could add a new data source just by adding a new ingestor service) and resilient (if the ingestor crashes, the API and frontend stay online). It also fulfills the "single-command local execution" requirement perfectly.

2. Database: PostgreSQL + TimescaleDB

Why (PostgreSQL): We needed a robust, open-source, and powerful database. PostgreSQL is a top-tier choice that is more feature-rich and scalable than alternatives like SQLite.

Why (TimescaleDB Extension): This is the key strategic choice for the backend. The assignment is fundamentally a time-series data problem.

Performance: TimescaleDB provides hypertables, which are specifically optimized for fast ingestion and querying of time-series data.

Sampling: The assignment requires sampling data into 1m and 5m bars. TimescaleDB provides continuous aggregates (materialized views) which do this inside the database automatically and far more efficiently than any Python script could.

3. Backend: FastAPI

Why: The assignment requires a Python-based backend.

Justification: FastAPI is a modern, high-performance web framework. It is asynchronous (async/await), making it perfect for an I/O-bound application that constantly waits for the database. It is significantly faster than traditional frameworks like Flask and includes automatic data validation and documentation, which is crucial for building clean APIs.

4. Frontend: Streamlit

Why: The assignment allows any frontend framework.

Justification: Streamlit is the fastest way to build a complex, interactive data dashboard entirely in Python.

Speed: We built a complete, interactive UI in ~200 lines of Python. Achieving the same with React or Angular would have required thousands of lines of code.

Features: It has all the required widgets (sidebar, file uploader, radio buttons, etc.) built-in.

Integration: It works seamlessly with Plotly (st.plotly_chart), which meets the requirement for interactive charts.

5. Analytics: pandas, statsmodels, numpy

Why: These libraries are the core of the Python quantitative analytics stack.

Justification:

pandas: Used for all data manipulation, such as pivoting price data to align timestamps.

statsmodels: Directly provides the exact statistical functions required: sm.OLS() (for OLS hedge ratio) and adfuller() (for the ADF test).

numpy: A core dependency for pandas and statsmodels, used for all low-level numerical calculations.

6. Charting: Plotly

Why: The assignment explicitly requires charts that support "zoom, pan, and hover."

Justification: Standard libraries like matplotlib produce static images. Plotly is designed for rich, interactive, web-based visualizations and integrates perfectly with Streamlit.

How to Run

Prerequisites

Docker Desktop must be installed and running.

Single-Command Setup

Clone this repository or ensure all files are in a single project folder.

Open a terminal in the project's root directory.

Run the following command:

docker-compose up --build


Wait for the services to build and start. The ingestor will begin collecting data.

Open your web browser and navigate to:

http://localhost:8501

The app may take 1-2 minutes to show data as the 1-minute aggregates are populated.

Dependencies

Backend (backend/requirements.txt)

websockets
psycopg2-binary
fastapi
uvicorn[standard]
pandas
statsmodels
numpy


Frontend (frontend/requirements.txt)

streamlit
requests
plotly
pandas
numpy
statsmodels