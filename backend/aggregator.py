import time
import psycopg2
import os
import logging

logging.basicConfig(level=logging.INFO)
DATABASE_URL = os.environ.get("DATABASE_URL")

def refresh_views():
    """Connects to the DB and refreshes the materialized views."""
    logging.info("Starting view refresh cycle...")
    sql_1m = "CALL refresh_continuous_aggregate('ohlc_1m', NULL, NULL);"
    sql_5m = "CALL refresh_continuous_aggregate('ohlc_5m', NULL, NULL);"

    conn = None  
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True 

        with conn.cursor() as curs:
            logging.info("Refreshing ohlc_1m...")
            curs.execute(sql_1m)
            logging.info("Refreshing ohlc_5m...")
            curs.execute(sql_5m)

        logging.info("Views refreshed successfully.")
    except Exception as e:
        logging.error(f"Error refreshing views: {e}")
    finally:
        if conn:
            conn.close()  
if __name__ == "__main__":
    while True:
        refresh_views()
        # Refresh every 60 seconds
        time.sleep(60)