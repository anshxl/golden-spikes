import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn.functional as F

# Focal Loss function
def focal_loss(logits, labels, gamma=2.0, alpha=None, class_weight=None):
    ce = F.cross_entropy(logits, labels, weight=class_weight, reduction="none")
    pt = torch.exp(-ce)                  # pt = p_t
    focal = (1 - pt)**gamma * ce         # modulating factor
    if alpha is not None:
        a = alpha[labels]                # class-specific alpha
        focal = a * focal
    return focal

# Custom Trainer Class
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        pw     = inputs.pop("phase_weight")        # shape: (batch,)
        outputs= model(**inputs)
        logits = outputs.logits
        # combine class-weighted cross-entropy + focal + phase weight
        losses = focal_loss(
            logits, labels,
            gamma=2.0,
            alpha=torch.tensor([1.0,1.0,1.0]).to(model.device),
            class_weight=self.class_weights
        )
        loss = (losses * pw).mean()
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
    train_weak['phase_weight'] = 0.1
    val_weak['phase_weight'] = 1.0

    train_llm, val_llm = train_test_split(llm_df,
                                            test_size=0.1,
                                            stratify=llm_df['llm_label'], 
                                            random_state=42)
    train_llm['phase_weight'] = 0.5
    val_llm['phase_weight'] = 1.0

    train_strong, test_strong = train_test_split(gold_df,
                                                test_size=0.1,
                                                stratify=gold_df['strong_label'],
                                                random_state=42)
    train_strong['phase_weight'] = 1.0
    test_strong['phase_weight'] = 1.0

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
        d.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'phase_weight'])

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
        num_train_epochs=5,
        seed=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(strong_ds['train'])):
        gold_train = strong_ds['train'].select(train_idx)
        gold_val = strong_ds['train'].select(val_idx)

        # Combine datasets
        combined_train = concatenate_datasets(weak_ds['train'], llm_ds['train'], gold_train).shuffle(seed=42)

        # Get class weights for the combined training set
        class_weights = get_class_weights(combined_train['label'], device)

        # Model + Trainer Setup
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)
        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

        # Freeze first 8 layers
        for name, param in model.named_parameters():
            if "layer." in name:
                layer_num = int(name.split("layer.")[1].split(".")[0])
                if layer_num < 8:
                    param.requires_grad = False
        print("Model layers frozen successfully!")

        # Set up differential optimizer
        head_prefixes = ('pre_classifier', 'classifier')

        decay_params, no_decay_params, head_params = [], [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(n.startswith(prefix) for prefix in head_prefixes):
                head_params.append(p)
                continue
            if any(nd in n for nd in ['bias', 'LayerNorm.weight']):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": 0.01,
                "lr": 5e-6,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": 5e-6,
            },
            {
                "params": head_params,
                "weight_decay": 0.0,
                "lr": 2e-5,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=TrainingArguments(
                **base_args,
                output_dir=f'models/fold_{fold}'
            ),
            optimizers=(optimizer, None),
            train_dataset=combined_train,
            eval_dataset=gold_val,
            compute_metrics=compute_f1,
            callbacks=[early_stopping]
        )
        print(f"Starting training for fold {fold}...")
        trainer.train()
        res = trainer.evaluate()
        fold_scores.append(res['eval_f1'])
        print(f"Fold {fold} F1: {res['eval_f1']:.4f}")
        # Save best model for this fold
        best_dir = f'models/fold_{fold}/best_model'
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)

    print(f"CV mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Get best fold
    best_fold = np.argmax(fold_scores)
    print(f"Best fold: {best_fold} with F1: {fold_scores[best_fold]:.4f}")
    
if __name__ == "__main__":
    main()
    print("Training script completed successfully!")