# Does Online Political Discourse Affect Gold Futures?

## Table of Contents

* [Research Question](#research-question)
* [Data Collection](#data-collection)
* [Data Processing](#data-processing)
* [Exploratory Analysis](#exploratory-analysis)
* [Sentiment Annotation & Modeling](#sentiment-annotation--modeling)
* [Analysis & Findings](#analysis--findings)
* [Project Structure](#project-structure)
* [Installation & Usage](#installation--usage)
* [Tech Stack](#tech-stack)
* [License](#license)

## Research Question

> **Does the tone and content of online political discussion on Reddit have a measurable relationship with movements in gold futures prices?**

I approach this as an exploratory study rather than a direct prediction exercise, seeking to understand if—and how—volatility in political sentiment aligns with market behavior.

## Data Collection

* **Subreddits**: `r/politicaldiscussion`, `r/geopolitics`, plus general finance communities for context.
* **Time Range**: Two years of posts and comments.
* **Reddit Data**: Ingested from PSAW monthly archives, cleaned, and stored in SQLite.
* **Gold Futures**: Daily OHLC from `yfinance`.

## Data Processing

1. **Cleaning & Preprocessing**

   * Drop removed/deleted entries.
   * Lowercase, strip URLs/mentions, normalize whitespace.
   * Merge by date and resample to daily frequency.
2. **Sentiment Metrics**

   * **Basic**: % positive / negative / neutral.
   * **Net Sentiment**: `%pos - %neg`.
   * **Volume Weighting**: scaled by daily comment count.
   * **Neutral Adjustment**: focus on non-neutral share.
   * **Volatility**: 7-day rolling std of net sentiment.
   * **Topic Filtering**: isolate comments mentioning "gold."

## Exploratory Analysis

* Stationarity tests (ADF) confirmed returns are stationary; sentiment required differencing.
* Correlation (Pearson/Spearman) at multiple horizons—no strong same‑day relationship.
* Rolling and cross‑correlation showed only negligible lead/lag signals.
* Granger causality up to 7 days returned no robust predictive effect.
* VAR impulse‑response analysis indicated gold returns moderately influence sentiment, not vice versa.

## Sentiment Annotation & Modeling

I first annotated our corpus in three tiers—weak labels (VADER), LLM labels, and a small gold standard—but instead of sequential phase-wise fine-tuning, I train one unified classifier via cross-validation on the gold set while leveraging the scale of the proxy labels. To generate LLM labels, I made a TinyLlama-powered annotation app that runs on llama-cpp.

1. **Annotation Tiers**  
   - **Weak Labels** (VADER): ~90% of data, tends toward neutral.  
   - **LLM Labels**: ~9% of data, more balanced across pos/neg/neu. 
   - **Gold Labels**: <1% of data, highest fidelity, skewed toward negative.  

2. **Cross-Validation Workflow**  
   - **5-Fold Gold CV**: Split the gold-standard set into five folds.  
   - For each fold:  
     - **Train set** = (all weak + all LLM) ∪ (gold_train fold)  
     - **Validation set** = gold_val fold (the most trusted labels)  
     - **Record** F1 on gold_val after training.  

3. **Weighting & Loss**  
   - **Phase weights**: 0.1 for weak, 0.5 for LLM, 1.0 for gold examples.  
   - **Class weights**: computed on the combined train set to correct imbalance.  
   - **Loss**: focal + weighted cross-entropy, with early stopping driven by gold_val F1.  

4. **Optimization Details**  
   - **Layer freezing**: first 8 transformer layers frozen for stability.  
   - **Differential LRs**: lower LR on frozen/base layers, higher LR on the classifier head.  

5. **Model Selection**  
   - After all folds complete, pick the checkpoint whose fold‐average gold_val F1 is highest.  
   - That “best” model becomes our final artifact for downstream inference.

This CV-centric approach ensures our tiniest, high-quality gold labels guide validation and selection, while the large proxy datasets supply the data volume needed to train a robust BERT classifier.

## Analysis & Findings

While the enhanced model yields more balanced and nuanced sentiment scores, our statistical tests still show minimal correlation or predictive power of political sentiment on gold returns.

**Value of Inquiry**: Even null results inform market‐sentiment research and highlight the challenges of extracting financial signals from social media.

## Project Structure

```
├── data/
│   ├── raw/            # Original archives
│   ├── processed/      # Cleaned daily metrics
│   └── annotations/    # phase1/2/3 CSVs
├── notebooks/          # EDA & correlation analyses
├── src/
│   ├── ingest/         # download & parse scripts
│   ├── preprocess/     # cleaning & resampling
│   ├── annotation/     # VADER, LLM, gold labeling
│   ├── modeling/       # train.py, inference.py
│   └── analysis/       # correlation notebook
├── models/             # fold checkpoints & final_best
├── requirements.txt
├── LICENSE
└── README.md           # ← this file
```

## Installation & Usage

1. **Clone**: `git clone <repo_url>`
2. **Env**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ingest data**:

   ```bash
   python src/ingest/load_zst.py --input data/raw --db data/comments.db
   ```
4. **Preprocess & label**:

   ```bash
   python src/preprocess/clean_and_resample.py
   python src/annotation/annotate.py --phase all
   ```
5. **Train**:

   ```bash
   python src/modeling/train.py  # runs CV and saves best
   ```
6. **Analyze correlations**:

   ```bash
   jupyter notebook notebooks/correlation_analysis.ipynb
   ```
7. **Inference**:

   ```bash
   python src/modeling/inference.py --model models/final_best
   ```

## Tech Stack

* **Core**: Python, Pandas, NumPy
* **NLP**: Hugging Face Transformers, Datasets, PyTorch
* **Stats**: SciPy, statsmodels
* **Storage**: SQLite
* **Visuals**: Matplotlib, Seaborn, Jupyter

## License

MIT © Your Name
