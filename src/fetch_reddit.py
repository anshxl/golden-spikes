import os
import time
from datetime import datetime
from dotenv import load_dotenv 
import praw
import pandas as pd
from sqlalchemy import create_engine, Table, Column, String, Integer, MetaData, Text, DateTime, insert

# Load credentials from .env file
load_dotenv()
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Setup SQL Engine
engine = create_engine("sqlite:///reddit_posts.db")
metadata = MetaData()

posts = Table(
    'posts_all', metadata,
    Column("id",            String,  primary_key=True),
    Column("subreddit",     String,  index=True),
    Column("title",         Text),
    Column("selftext",      Text),
    Column("created_utc",   Integer, index=True),
    Column("subscribers",   Integer),
)
metadata.create_all(engine)

# Helper function 
def fetch_window(sub: praw.models.Subreddit, start_ts: int, end_ts: int):
    """
    Yields submissions from `sub` with created_utc in [start_ts, end_ts].
    """
    query = f"timestamp:[{start_ts} TO {end_ts}]"
    for post in sub.search(
        query,
        sort="new",
        syntax="lucene",
        time_filter="all",
        limit=None
    ):
        yield {
            "id":           post.id,
            "subreddit":    sub.display_name,
            "title":        post.title,
            "selftext":     post.selftext or "",
            "created_utc":  int(post.created_utc),
            "subscribers":  getattr(sub, "subscribers", None)
        }

# Function to fetch posts from a subreddit
def fetch_and_store(    sub_names: list[str],
    days_back: int = 730,
    chunk_days: int = 30,
    batch_size: int = 5_000
):
    cutoff_ts = int(time.time()) - days_back * 24*3600
    now_ts    = int(time.time())
    chunk_delta = chunk_days * 24*3600

    for sub_name in sub_names:
        sub = reddit.subreddit(sub_name)
        print(f"\nFetching r/{sub_name}, subscribers ~ {sub.subscribers}")

        buffer: list[dict] = []
        window_end = now_ts

        # loop windows: [window_start, window_end]
        while window_end > cutoff_ts:
            window_start = max(cutoff_ts, window_end - chunk_delta)
            print(f"Window {datetime.fromtimestamp(window_start)} → {datetime.fromtimestamp(window_end)}")

            for record in fetch_window(sub, window_start, window_end):
                print(record)
                # Check output data type
                if not isinstance(record, dict):
                    print(record)
                    raise TypeError(f"Expected dict, got {type(record)}")
 
                # Check required fields
                required_fields = {"id", "subreddit", "title", "selftext", "created_utc", "subscribers"}
                if not required_fields.issubset(record.keys()):
                    raise ValueError(f"Missing required fields in record: {record.keys()}")

                buffer.append(record)
                # flush in batches
                if len(buffer) >= batch_size:
                    _flush_buffer(buffer)
            window_end = window_start
            time.sleep(1)  # gentle rate limiting
        # flush remaining records
        if buffer:
            _flush_buffer(buffer)

# Helper function to flush the buffer to the database
def _flush_buffer(buffer: list[dict]):
    """
    Flush the buffer to the database
    """
    df = pd.DataFrame(buffer)
    with engine.begin() as conn:
        stmt = insert(posts).prefix_with("OR IGNORE")
        conn.execute(stmt, df.to_dict(orient="records"))
    print(f"Flushed {len(buffer)} records → posts_ext")
    buffer.clear()

if __name__ == "__main__":
    TARGET_SUBS = [
        "Gold","Investing","Finance","Economics","WallStreetBets",
        "PoliticalDiscussion","PoliticalEconomy","USPolitics",
        "worldpolitics","Geopolitics"
    ]
    fetch_and_store(TARGET_SUBS)