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
        torch_dtype=torch.float16
    )

    results = pipe(df['clean_body'].tolist(), batch_size=32, 
                   truncation=True, padding=True, max_length=512)
    
    df['pred_label'] = [ {"LABEL_0":0, "LABEL_1":1, "LABEL_2":2}[result['label']] for result in results ]
    df['pred_score'] = [ result['score'] for result in results ]

    # Save df
    df.to_csv('data/predictions.csv', index=False)