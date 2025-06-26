import os
from dotenv import load_dotenv 
import praw
import pandas as pd
from sqlalchemy import create_engine, Table, Column, String, Integer, MetaData, Text, DateTime

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
    'posts', metadata,
    Column("id", String, primary_key=True),
    Column("subreddit", String, index=True),
    Column("title", Text),
    Column("selftext", Text),
    Column("created_utc", Integer, index=True),
)
metadata.create_all(engine)

# Function to fetch posts from a subreddit
def fetch_and_store(subreddits, limit=500):
    records = []
    for sub in subreddits:
        for submission in reddit.subreddit(sub).new(limit=limit):
            post_data = {
                "id": submission.id,
                "subreddit": sub,
                "title": submission.title,
                "selftext": submission.selftext or "",
                "created_utc": int(submission.created_utc)
            }
            records.append(post_data)
    df = pd.DataFrame(records)
    df.to_sql('posts', con=engine, if_exists='append', index=False)

if __name__ == "__main__":
    subreddits = ["Gold", "WallStreetBets", "PoliticalDiscussion"]
    print("Fetching posts from subreddits:", subreddits)
    fetch_and_store(subreddits, limit=100)  # Fetch 100 posts from each subreddit
    print("Data fetched and stored successfully.")