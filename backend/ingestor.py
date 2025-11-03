import asyncio
import websockets
import json
import psycopg2
import os
import logging
from contextlib import contextmanager
from datetime import datetime  

logging.basicConfig(level=logging.INFO)

# --- Database Connection ---
DATABASE_URL = os.environ.get("DATABASE_URL")

@contextmanager
def get_db_connection():
    """Provides a database connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    finally:
        conn.close()

def insert_tick(conn, tick_data):
    """Inserts a single tick into the database."""
    sql = "INSERT INTO ticks (timestamp, symbol, price, size) VALUES (%s, %s, %s, %s)"
    try:
        # Convert millisecond timestamp to a datetime object
        ts_datetime = datetime.fromtimestamp(tick_data['ts'] / 1000.0) 

        with conn.cursor() as curs:
            curs.execute(sql, (
                ts_datetime,  # <-- PASS THE CONVERTED DATETIME OBJECT
                tick_data['symbol'],
                tick_data['price'],
                tick_data['size']
            ))
        conn.commit()
    except Exception as e:
        logging.error(f"Error inserting tick: {e}")
        conn.rollback()
        
# --- WebSocket Client ---
def normalize_tick(symbol, msg):
    """Normalizes a trade message from Binance."""
    return {
        "ts": msg['T'],      # Trade time (timestamp)
        "symbol": symbol.upper(),
        "price": float(msg['p']),  # Price
        "size": float(msg['q'])    # Size/Quantity
    }

async def ws_client(symbols, db_conn):
    """Main WebSocket client logic."""
    # Format symbols for the stream URL
    streams = [f"{sym.lower()}@trade" for sym in symbols]
    url = "wss://fstream.binance.com/ws/" + "/".join(streams)
    logging.info(f"Connecting to: {url}")

    try:
        async with websockets.connect(url) as ws:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                # Check for trade data
                if data.get('e') == 'trade':
                    symbol = data['s']
                    tick = normalize_tick(symbol, data)
                    insert_tick(db_conn, tick)
                    # Optional: Log every 100th tick to avoid spam
                    if int(tick['ts']) % 100 == 0:
                        logging.info(f"Ingested tick for {symbol}: {tick['price']}")
                        
    except Exception as e:
        logging.error(f"WebSocket error: {e}. Reconnecting...")
        await asyncio.sleep(5) # Wait before reconnecting

async def main():
    symbols_to_track = ["BTCUSDT", "ETHUSDT"] # Add more symbols here
    
    # Keep the connection open
    with get_db_connection() as conn:
        await ws_client(symbols_to_track, conn)

if __name__ == "__main__":
    asyncio.run(main())