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
N_PER_CELL = 1200

# def stratified_sample(df, source_label):
#     samples = []
#     for sub in TARGET_SUBS:
#         for lbl in LABELS:
#             cell = df[(df.subreddit==sub) & (df.weak_label==lbl)]
#             if cell.empty:
#                 continue
#             n = min(len(cell), N_PER_CELL)
#             samp = cell.sample(n, random_state=42)
#             samp["source"] = source_label
#             samples.append(samp)
#     return pd.concat(samples, ignore_index=True)

# posts_sample = stratified_sample(posts, "posts")
# comments_sample = stratified_sample(comments, "comments")

# # Combine samples
# gold_df = pd.concat([posts_sample, comments_sample], ignore_index=True)

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
