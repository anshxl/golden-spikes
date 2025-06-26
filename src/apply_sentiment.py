import os

def main():
    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import pipeline
    from sqlalchemy import create_engine
    import torch

    # Set up the database connection
    DB_PATH = "sqlite:///reddit_posts.db"
    engine = create_engine(DB_PATH)

    # Load cleaned data from the database
    df = pd.read_sql("posts_clean", engine)
    print(f"Loaded {len(df)} posts from the database.")

    # Initialize the sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device= -1
    )

    # Apply in batches
    results = []
    batch_size = 32
    for i  in tqdm(range(0, len(df), batch_size), desc="Sentiment Batches"):
        batch = df.iloc[i:i + batch_size]
        texts = (batch['title'] + ". " + batch['selftext']).tolist()
        outs = sentiment_pipeline(texts, truncation=True)
        for idx, out in enumerate(outs):
            results.append({
                "id":          df.at[idx, "id"],
                "date":        df.at[idx, "date"],
                "label":       out["label"],        # “POSITIVE” or “NEGATIVE”
                "score":       float(out["score"])  # confidence
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to the database
    results_df.to_sql("posts_sentiment", engine, if_exists="replace", index=False)
    print(f"Saved sentiment analysis results for {len(results_df)} posts to the database.")

if __name__ == "__main__":
    main()