import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def main():
    # Load data
    df = pd.read_csv('data/full_comments.csv')

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/final_model')
    model = AutoModelForSequenceClassification.from_pretrained('models/final_model')
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        torch_dtype=torch.float16,
        top_k=None
    )

    results = pipe(df['clean_body'].tolist(), batch_size=32, 
                   truncation=True, padding=True, max_length=512)
    
    # 1) Build a DataFrame of scores per label
    score_df = pd.DataFrame([
        { d['label']: d['score'] for d in res }
        for res in results
    ])

    # 2) Map LABEL_0/1/2 → your 0/1/2 coding (if you need numeric labels)
    label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

    # 3) Pick the max‐score label
    score_df['pred_label'] = score_df.idxmax(axis=1).map(label2id)
    score_df['pred_score'] = score_df.max(axis=1)

    # Merge and save
    out = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # Save df
    df.to_csv('data/predictions.csv', index=False)