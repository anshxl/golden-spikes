import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch

print("Torch sees CUDA:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())

# Load in the datasets
weak_df = pd.read_csv('data/phase1/weak_label.csv')
llm_df = pd.read_csv('data/phase2/llm_label.csv')
gold_df = pd.read_csv('data/phase3/strong_label_cleaned.csv')

# Lower case label columns
weak_df['weak_label'] = weak_df['weak_label'].str.lower()
llm_df['llm_label'] = llm_df['llm_label'].str.lower()
gold_df['strong_label'] = gold_df['strong_label'].str.lower()

# Create label columns
label2id = {"negative": 0, "neutral": 1, "positive": 2}
weak_df['label'] = weak_df['weak_label'].map(label2id)
llm_df['label'] = llm_df['llm_label'].map(label2id)
gold_df['label'] = gold_df['strong_label'].map(label2id)

# Train test split
train_weak, val_weak = train_test_split(weak_df, 
                                        test_size=0.1,
                                        stratify=weak_df['weak_label'], 
                                        random_state=42)
train_llm, val_llm = train_test_split(llm_df,
                                        test_size=0.1,
                                        stratify=llm_df['llm_label'], 
                                        random_state=42)
train_strong, temp_strong = train_test_split(gold_df,
                                            test_size=0.2,
                                            stratify=gold_df['strong_label'],
                                            random_state=42)
val_strong, test_strong = train_test_split(temp_strong,
                                            test_size=0.5,
                                            stratify=temp_strong['strong_label'],
                                            random_state=42)

# Create HF datasets
weak_ds = DatasetDict({
    'train': Dataset.from_pandas(train_weak),
    'validation': Dataset.from_pandas(val_weak)
})
llm_ds = DatasetDict({
    'train': Dataset.from_pandas(train_llm),
    'validation': Dataset.from_pandas(val_llm)
})
strong_ds = DatasetDict({
    'train': Dataset.from_pandas(train_strong),
    'validation': Dataset.from_pandas(val_strong),
    'test': Dataset.from_pandas(test_strong)
})
print("Datasets created successfully!")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_fn(batch):
    return tokenizer(
        batch['clean_body'],
        padding='max_length',
        truncation=True
    )

# Tokenize datasets
weak_ds = weak_ds.map(tokenize_fn, batched=True)
llm_ds = llm_ds.map(tokenize_fn, batched=True)
strong_ds = strong_ds.map(tokenize_fn, batched=True)
print("Datasets tokenized successfully!")

# Set format for PyTorch
for d in [weak_ds, llm_ds, strong_ds]:
    d.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Model + Trainer Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)

base_args = dict(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    fp16=True,
    no_cuda=False,
    gradient_accumulation_steps=1,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    seed=42
)

# Phase 1
args1 = TrainingArguments(output_dir='models/phase1',
                          learning_rate=2e-5,
                          num_train_epochs=1,
                          **base_args)
trainer1 = Trainer(model=model,
                   args=args1,
                   train_dataset=weak_ds['train'],
                   eval_dataset=weak_ds['validation'],
)
# Confirm trainer device
print("Trainer will run on:", trainer1.args.device)
trainer1.train()

# Drop weak label dataset from memory
del weak_ds

#Save the model after phase 1
model.save_pretrained('models/phase1')
print("Phase 1 training complete!")

# Phase 2
args2 = TrainingArguments(output_dir='models/phase2',
                          learning_rate=1e-5,
                          num_train_epochs=1,
                          **base_args)
trainer2 = Trainer(model=model,
                   args=args2,
                   train_dataset=llm_ds['train'],
                   eval_dataset=llm_ds['validation'])
trainer2.train()
# Save the model after phase 2
model.save_pretrained('models/phase2')
print("Phase 2 training complete!")

# Phase 3
args3 = TrainingArguments(output_dir='models/phase3',
                          learning_rate=1e-6,
                          num_train_epochs=3,
                          **base_args)
trainer3 = Trainer(model=model,
                    args=args3,
                    train_dataset=strong_ds['train'],
                    eval_dataset=strong_ds['validation'])
trainer3.train()
# Save the model after phase 3
model.save_pretrained('models/phase3')
print("Phase 3 training complete!")

# Final evaluation on test set
test_results = trainer3.evaluate(strong_ds['test'])
print("Test Results:", test_results)

# Save the evaluation results
with open('models/phase3/evaluation_results.txt', 'w') as f:
    f.write(str(test_results))

# Save the final model
model.save_pretrained('models/final_model')
print("Final model saved successfully!")


