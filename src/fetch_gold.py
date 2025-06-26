import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Float, MetaData, Date

# Set up SQL Engine
engine = create_engine("sqlite:///reddit_posts.db")
metadata = MetaData()

# Define the gold prices table
gold = Table(
    "gold_prices", metadata,
    Column("date", Date, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", Float),
)
metadata.create_all(engine)

# Function to fetch gold prices
def fetch_and_store(
        ticker: str = "GC=F",
        period: str = "2y",
        interval: str = "1d",
):
    df = yf.download("GC=F", period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    df = df.reset_index()

    # Assert that the DataFrame has the expected columns
    expected_columns = ["date", "Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in expected_columns):
        print("DataFrame columns:", df.columns)
        raise ValueError(f"DataFrame is missing expected columns: {expected_columns}")
    else:
        print("DataFrame has all expected columns.")
    # Rename columns to match the table schema
    # df2 = df.rename(columns={
    #     "('date', '')": "date",
    #     "('Close', 'GC=F')": "close",
    #     "('High', 'GC=F')": "high",
    #     "('Low', 'GC=F')": "low",
    #     "('Open', 'GC=F')": "open",
    #     "('Volume', 'GC=F')": "volume",
    # })
    df.to_sql("gold", con=engine, if_exists="replace", index=False)
    print(f"Stored {len(df)} records in the gold_prices table for {ticker} ticker.")

if __name__ == "__main__":
    print("Fetching gold prices...")
    fetch_and_store()
    print("Gold prices fetched and stored successfully.")
