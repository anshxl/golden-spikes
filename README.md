# Sentiment Analysis of Reddit Comments and Gold Futures Correlation

## Table of Contents

* [Project Overview](#project-overview)
* [Data Collection](#data-collection)
* [Data Processing](#data-processing)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Sentiment Labeling & Annotation](#sentiment-labeling--annotation)
* [Model Fine‑Tuning](#model-fine‑tuning)
* [Future Work](#future-work)
* [Project Structure](#project-structure)
* [Installation & Usage](#installation--usage)
* [Tech Stack](#tech-stack)
* [License](#license)

## Project Overview

This project aims to analyze sentiment in Reddit comments across multiple finance‑related subreddits and explore correlations with gold futures prices. It covers end‑to‑end data ingestion, preprocessing, exploratory analysis, annotation, and transformer‑based model fine‑tuning. Downstream applications include predictive trading models and broader sentiment monitoring.

## Data Collection

* **Subreddits**: `r/investing`, `r/gold`, `r/politicaldiscussion`, `r/geopolitics`, `r/finance`
* **Time Range**: Last 2 years
* **Method**:

  1. Initial approach using PRAW (limited to 1,000 results per query).
  2. Migrated to monthly subreddit dumps via PSAW torrent streams.
  3. Downloaded Zstandard (`.zst`) archives with Transmission.
  4. Ingested posts and comments into SQLite databases with custom Python scripts.
* **Financial Data**: Retrieved gold futures OHLC data using `yfinance`.

## Data Processing

1. **Cleaning**:

   * Removed `[removed]` and `[deleted]` entries.
   * Combined post titles and bodies; kept comments intact.
   * Stripped mentions, URLs, HTML entities; normalized whitespace; lowercased text.
2. **Storage**:

   * Cleaned data stored in SQLite tables for easy querying.

## Exploratory Data Analysis (EDA)

* Analyzed volume trends: daily & weekly post/comment counts.
* Examined text lengths (character distributions).
* Extracted top n‑grams (via NLTK vs. `CountVectorizer`).
* Computed inter‑subreddit correlation matrices.
* Generated VADER sentiment scores for weak labeling:

  * `score ≥ 0.7` → positive
  * `score ≤ -0.7` → negative
  * `|score| ≤ 0.1` → neutral
* **Findings**:

  * Posts largely neutral (question‑style); comments showed richer sentiment.
  * Dropped posts and non‑discussion subreddits (`r/investing`, `r/finance`, `r/gold` advice‑seeking) from further analysis.
  * Final dataset: \~1.7M comments.

## Sentiment Labeling & Annotation

* **Weak Labels**: VADER labels skewed neutral, insufficient nuance.
* **LLM Annotation**:

  * Sampled 100K representative comments.
  * Built a TinyLlama annotator using `llama-cpp` for cost‑effective inference.
  * Generated balanced LLM labels (positive, negative, neutral).
* **Gold Standard**:

  * Manually annotated 1,500 comments for final evaluation.

## Model Fine‑Tuning (In Progress)

Three‐phase training strategy using Hugging Face Transformers:

1. **Phase I**: Weak labels — 1 epoch, lr=2e‑5
2. **Phase II**: LLM labels — 1 epoch, lr=1e‑5
3. **Phase III**: Manual labels — 3 epochs, lr=5e‑6

## Future Work

* Use fine‑tuned model to score full comment corpus.
* Analyze sentiment–gold futures correlations.
* Develop a predictive trading signal based on sentiment trends.
* Extend model to other subreddits or sentiment tasks (political, economic, social).

## Project Structure

```
├── data/
│   ├── phase1/           # Weak labels
│   ├── phase2/           # LLM Labels
│   └── phase3/           # Gold Standard Labels
├── notebooks/            # EDA & analysis
├── src/
│   ├── fetch_gold/        # yfinance API
│   ├── fetch_torrent/     # ZST -> SQLite
│   ├── prep_data/         # Basic preprocessing
│   ├── train/             # Training & evaluation  
├── models/               # Checkpoints & logs             
├── requirements.txt
├──sentiment.sb            # SLURM Script
└── README.md
```

## Installation & Usage

1. **Clone repo**: `git clone <repo_url>`
2. **Setup environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Data ingestion**:

   ```bash
   python src/ingestion/load_zst.py --input data/raw --db data/sqlite/comments.db
   ```
4. **Preprocessing & EDA**:

   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```
5. **Annotation**:

   ```bash
   python src/annotation/llm_annotator.py --sample 100000
   ```
6. **Training**:

   ```bash
   python src/modeling/train.py --phase 1
   ```

## Tech Stack

* **Language & Env**: Python, Jupyter, `venv`
* **Data**: SQLite, `yfinance`
* **NLP & Modeling**: PyTorch, Hugging Face Transformers & Datasets, `llama-cpp`
* **EDA & Visualization**: scikit‑learn, VADER, Matplotlib, NLTK
* **Version Control**: Git

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
