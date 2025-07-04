```markdown
# Reddit Sentiment vs. Gold-Price Signal Project

## 🧭 Project Overview

We’re building an end-to-end NLP pipeline on a ~2 GB Reddit comments corpus to surface sentiment/volume signals that may correlate with gold‐price movements (and beyond). So far we have:

1. **Environment Setup**  
2. **Data Collection & Ingestion**  
3. **Exploratory Data Analysis**  
4. **Weak‐Labeling with VADER**  
5. **Small-Scale LLM Annotation & Comparison**  
6. **Side-Project: Large-Scale LLM Annotation on HPC**  
7. **Next Steps & Potential Pivots**

---

## 1. Environment & Virtualenv

```bash
# Create & activate
python3.11 -m venv reddit-env
source reddit-env/bin/activate

# Core dependencies
pip install \
  praw sqlalchemy pandas numpy matplotlib \
  zstandard ftfy nltk vaderSentiment \
  transformers torch langchain llama-cpp-python
```

* **NLTK** data:

  ```python
  import nltk
  nltk.download("punkt")
  nltk.download("stopwords")
  ```

---

## 2. Data Collection & Ingestion

### 2.1 PRAW → API Limit Hit

* Attempted time-windowed `sub.search(syntax='lucene', time_filter='all')` → 1 000-post cap, no historical backfill.

### 2.2 Pushshift REST → Moderator-Only 404s

* PSAW wrapper failed due to stale endpoints and 404s.

### 2.3 Official Pushshift Monthly Dumps → Torrent

* Switched to streaming `.zst` from Academic Torrents.
* **Torrent selective download** by file-priority (GUI or `libtorrent`) to grab only:

  ```
  Economics_submissions.zst
  Investing_submissions.zst
  Finance_submissions.zst
  Gold_submissions.zst
  WallStreetBets_submissions.zst
  ```
* **Ingestion script**: line-by-line zstd → batch-insert SQLite `reddit_posts.db`.

---

## 3. Basic EDA

* **Counts & Date Ranges**: posts vs. comments per subreddit.
* **Time-Series Volume**: daily posts/comments, rolling-30d averages.
* **Text Length Distributions**: char-count histograms.
* **Token Frequency**: unigrams, bigrams, trigrams after NLTK stop-word removal.
* **VADER Baseline**:

  * Self-text → \~85 % neutral
  * Titles → \~45 % neutral
  * Comments → \~25.8 % neutral
* **Subreddit Breakdown**: proportions of neg/neu/pos.

---

## 4. Weak-Labeling with VADER

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
comments["vader_score"] = comments["clean_body"].apply(lambda t: sid.polarity_scores(t)["compound"])
```

* **Strong labels**:

  * Positive: `score ≥ +0.7`
  * Negative: `score ≤ –0.7`
  * Neutral: `|score| ≤ 0.2`
* Filtered `comments_strong` table saved for sampling.

---

## 5. Small-Scale LLM Annotation & Comparison

* Sampled \~7 200 comments stratified by subreddit/weak\_label.
* Used **GPT-4** via LangChain to label **text\_type** (Opinion/Question/Other) and **sentiment**.
* **Confusion Matrix** vs. VADER showed VADER overestimates positivity (thanks-signoffs, generic lexicon misses finance nuance).
* LLM labels far heavier on “Negative” for market commentary → more domain-aware.

---

## 6. Side-Project: Large-Scale LLM Annotation on HPC

We’re launching a separate repo to:

1. **Extract** `comments_cleaned` → `comments_to_annotate.csv`
2. **LangChain + LlamaCpp** GPU pipeline

   * `n_gpu_layers`, `n_batch` for GPU acceleration
   * `SQLiteCache` for memoization
   * `ThreadPoolExecutor` for parallelism
3. **Annotate millions** of comments efficiently
4. **Output** `comments_llm_annotated.csv` for downstream modeling

---

## 7. Next Steps & Potential Pivots

1. **Model Training**

   * Fine-tune a small Transformer head on:

     * LLM‐pseudo-labels (weak pre-train)
     * Human gold-set (strong fine-tune)
2. **Daily Feature Aggregation**

   * `comment_count`, `avg_sentiment`, `% positive`, rolling stats
3. **Econometric Testing**

   * Correlation, Granger causality, ARIMAX on gold returns
4. **Prototype Dashboard & Alerts**

   * Streamlit or Dash
5. **Potential Pivots**

   * Apply the same pipeline to other asset classes or macro topics
   * Topic modeling on the corpus for broader socio-economic insights

---

**We have a robust ingestion & EDA foundation, validated our weak-label vs. LLM pipeline, and are now scaling LLM annotation on HPC to power our production NLP model—anchored to gold price insights but flexible for new directions.**
