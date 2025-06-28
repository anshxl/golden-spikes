import pandas as pd
import argparse
from sqlalchemy import create_engine

engine = create_engine("sqlite:///reddit_posts.db")

def clean_reddit(table: str = "posts_archive"):
    # Load raw data
    df = pd.read_sql(table, con=engine)

    # Convert 'created_utc' to datetime
    df["date"] = pd.to_datetime(df["created_utc"], unit='s').dt.date

    # Drop duplicates
    df = df.drop_duplicates(subset=["id"])

    # Keep only needed cols
    if table.startswith("comments"):
        df = df[["id", "subreddit", "parent_id", "body", "date"]]
    else:
        df = df[["id", "subreddit", "title", "selftext", "date"]]
    
    # Overwrite or create a cleaned table
    clean_name = table.replace("_archive", "_clean")
    df.to_sql(clean_name, con=engine, if_exists='replace', index=False)
    print(f"Cleaned {table} -> {clean_name}: ({len(df)} rows).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Reddit posts data.")
    parser.add_argument("--table", type=str, default="posts_archive", help="Table name to clean")
    args = parser.parse_args()
    table_name = args.table
    print("Cleaning Reddit posts...")
    clean_reddit(table_name)
    print("Reddit posts cleaned successfully.")