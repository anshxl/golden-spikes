import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///reddit_posts.db")

def clean_reddit():
    # Load raw data
    df = pd.read_sql("posts", con=engine)

    # Convert 'created_utc' to datetime
    df["date"] = pd.to_datetime(df["created_utc"], unit='s').dt.date

    # Drop duplicates
    df = df.drop_duplicates(subset=["id"])

    # Keep only needed cols
    df = df[["id", "subreddit", "title", "selftext", "date"]]
    
    # Overwrite or create a cleaned table
    df.to_sql("posts_clean", engine, if_exists="replace", index=False)
    print(f"Cleaned posts → posts_clean ({len(df)} rows).")

if __name__ == "__main__":
    print("Cleaning Reddit posts...")
    clean_reddit()
    print("Reddit posts cleaned successfully.")