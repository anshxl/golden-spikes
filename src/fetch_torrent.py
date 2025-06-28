import os
import json
import glob
import io
import time
import zstandard as zstd
import pandas as pd
import logging

from sqlalchemy import create_engine, Table, Column, String, Integer, MetaData, Text, insert

# Config
DATA_DIR = "subreddits24"
TARGET_SUBS = [
        "Gold","investing","finance","Economics","wallstreetbets",
        "PoliticalDiscussion","geopolitics"
]
CUT_OFF_TS  = int(time.time() - 6*365*24*3600)      # two years ago (optional)
BATCH_SIZE = 5_000
LOG_INTERVAL = 100_000

# Setup logging
log = logging.getLogger("ingest")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Database setup
engine = create_engine("sqlite:///reddit_posts.db")
metadata = MetaData()

posts_table = Table(
    "posts_archive", metadata,
    Column("id",          String, primary_key=True),
    Column("subreddit",   String, index=True),
    Column("title",       Text),
    Column("selftext",    Text),
    Column("created_utc", Integer, index=True),
)

comments_table = Table(
    "comments_archive", metadata,
    Column("id",          String, primary_key=True),
    Column("subreddit",   String, index=True),
    Column("parent_id",   String),
    Column("body",        Text),
    Column("created_utc", Integer, index=True),
)

metadata.create_all(engine)

# Stream-read util
def read_lines_zst(path):
    """
    Yield one UTF-8 decoded line at a time from a .zst file,
    skipping any decode errors.
    """
    with open(path, "rb") as fh:
        # 1. Wrap the file handle in the zstd decompressor
        dctx = zstd.ZstdDecompressor().stream_reader(fh)
        # 2. Wrap that in a text wrapper that gives us lines
        text_stream = io.TextIOWrapper(dctx, encoding="utf-8", errors="ignore")
        # 3. Iterate
        for line in text_stream:
            yield line.rstrip("\n")
        # 4. Clean up
        text_stream.close()
        dctx.close()

# Flush buffer helper
def flush_buffer(buf, table):
    df = pd.DataFrame(buf)
    with engine.begin() as conn:
        stmt = insert(table).prefix_with("OR IGNORE")
        conn.execute(stmt, df.to_dict(orient="records"))
    buf.clear()

# Ingest function
def ingest_all_zst():
    pattern = os.path.join(DATA_DIR, "*_*.zst")
    files   = sorted(glob.glob(pattern), reverse=True)

    for path in files:
        name = os.path.basename(path)
        sub, kind = name.split("_", 1)  # e.g. "Economics", "submissions.zst" or "comments.zst"
        kind = kind.split(".zst")[0]    # "submissions" or "comments"

        if sub not in TARGET_SUBS:
            log.info(f"Skipping {name}: not in TARGET_SUBS")
            continue

        # pick the right table & fields
        if kind == "submissions":
            table    = posts_table
            field_map= lambda post: {
                "id":          post["id"],
                "subreddit":   post["subreddit"],
                "title":       post.get("title",""),
                "selftext":    post.get("selftext",""),
                "created_utc": int(post.get("created_utc",0))
            }
        elif kind == "comments":
            table    = comments_table
            field_map= lambda post: {
                "id":          post["id"],
                "subreddit":   post["subreddit"],
                "body":        post.get("body",""),
                "created_utc": int(post.get("created_utc",0))
            }
        else:
            log.warning(f"Unknown kind `{kind}` in file {name}, skipping.")
            continue

        log.info(f"Ingesting {kind} for r/{sub} from {name}")
        buffer     = []
        line_count = 0
        for line in read_lines_zst(path):
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            ts = int(post.get("created_utc",0))
            if ts < CUT_OFF_TS:
                continue

            buffer.append(field_map(post))
            line_count += 1

            if len(buffer) >= BATCH_SIZE:
                flush_buffer(buffer, table)

            if line_count % LOG_INTERVAL == 0:
                log.info(f"{sub} {kind}: processed {line_count:,} lines")

        if buffer:
            flush_buffer(buffer, table)
        log.info(f"Finished {sub} {kind}: {line_count:,} lines ingested")

    log.info("All files processed.")

if __name__ == "__main__":
    start_time = time.time()
    ingest_all_zst()
    log.info(f"Total time taken: {time.time() - start_time:.2f} seconds")