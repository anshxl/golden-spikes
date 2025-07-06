import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np
import torch

# Custom Trainer Class
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
# -- Helper Functions -- #
def get_class_weights(labels, device):
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)

def compute_f1(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="weighted")
    }

# -- Main Function -- #
def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Load datasets
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


    # Get class weights
    weak_class_weights = get_class_weights(train_weak['label'], device)
    llm_class_weights = get_class_weights(train_llm['label'], device)
    strong_class_weights = get_class_weights(train_strong['label'], device)

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

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    def _tokenize_fn(batch):
        return tokenizer(
            batch['clean_body'],
            padding='max_length',
            truncation=True
    )
    # Tokenize datasets
    weak_ds = weak_ds.map(_tokenize_fn, batched=True)
    llm_ds = llm_ds.map(_tokenize_fn, batched=True)
    strong_ds = strong_ds.map(_tokenize_fn, batched=True)
    print("Datasets tokenized successfully!")

    # Set format for PyTorch
    for d in [weak_ds, llm_ds, strong_ds]:
        d.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Model + Trainer Setup
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    base_args = dict(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        fp16=True,
        no_cuda=False,
        gradient_accumulation_steps=1,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        seed=42
    )

    # Phase 1
    args1 = TrainingArguments(output_dir='models/phase1',
                            learning_rate=2e-5,
                            num_train_epochs=3,
                            **base_args)
    trainer1 = WeightedTrainer(class_weights=weak_class_weights,
                    model=model,
                    args=args1,
                    train_dataset=weak_ds['train'],
                    eval_dataset=weak_ds['validation'],
                    compute_metrics=compute_f1,
                    callbacks=[early_stopping]
    )
    # Confirm trainer device
    print("Trainer will run on:", trainer1.args.device)
    trainer1.train()

    # Drop weak label dataset from memory
    del weak_ds

    #Save the model after phase 1
    tokenizer.save_pretrained('models/phase1')
    trainer1.save_model('models/phase1')
    print("Phase 1 training complete!")

    # Phase 2
    args2 = TrainingArguments(output_dir='models/phase2',
                            learning_rate=1e-5,
                            num_train_epochs=3,
                            weight_decay=0.01,
                            warmup_ratio=0.1,
                            lr_scheduler_type='cosine',
                            **base_args)
    trainer2 = WeightedTrainer(class_weights=llm_class_weights,
                    model=model,
                    args=args2,
                    train_dataset=llm_ds['train'],
                    eval_dataset=llm_ds['validation'],
                    compute_metrics=compute_f1,
                    callbacks=[early_stopping])
    trainer2.train()

    # Save the model after phase 2
    trainer2.save_model('models/phase2')
    tokenizer.save_pretrained('models/phase2')
    print("Phase 2 training complete!")

    # Phase 3
    args3 = TrainingArguments(output_dir='models/phase3',
                            learning_rate=1e-6,
                            num_train_epochs=5,
                            weight_decay=0.01,
                            warmup_ratio=0.1,
                            lr_scheduler_type='cosine',
                            **base_args)
    trainer3 = WeightedTrainer(class_weights=strong_class_weights,
                        model=model,
                        args=args3,
                        train_dataset=strong_ds['train'],
                        eval_dataset=strong_ds['validation'],
                        compute_metrics=compute_f1,
                        callbacks=[early_stopping])
    trainer3.train()

    # Save the model after phase 3
    trainer3.save_model('models/phase3')
    tokenizer.save_pretrained('models/phase3')
    print("Phase 3 training complete!")

    # Final evaluation on test set
    test_results = trainer3.evaluate(strong_ds['test'])
    print("Test Results:", test_results)

    # Save the evaluation results
    with open('models/phase3/evaluation_results.txt', 'w') as f:
        f.write(str(test_results))

    # Save the final model
    trainer3.save_model('models/final_model')
    tokenizer.save_pretrained('models/final_model')
    print("Final model saved successfully!")

if __name__ == "__main__":
    main()
    print("Training script completed successfully!")