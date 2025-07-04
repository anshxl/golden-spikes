import os
import pandas as pd
from sqlalchemy import create_engine

# Connect to the database
engine = create_engine('sqlite:///reddit_posts.db')

# Load data
# posts = pd.read_sql("posts_labeled", engine)
comments = pd.read_sql("comments_labeled", engine)

# Define strata and sample size
TARGET_SUBS = ['geopolitics','PoliticalDiscussion']
LABELS = ['positive', 'negative', 'neutral']
N_PER_CELL = 16000

# Create a gold standard dataset from comments
samples=[]
for sub in TARGET_SUBS:
  for lbl in LABELS:
    cell = comments[(comments.subreddit==sub)&(comments.weak_label==lbl)]
    samples.append(cell.sample(min(len(cell),N_PER_CELL), random_state=42))

# Save the gold standard dataset
gold_df = pd.concat(samples)
gold_df.to_csv("data/gold_standard.csv", index=False)

print(f"Exported {len(gold_df)} samples for manual labeling.")
