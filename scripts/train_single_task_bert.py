import os
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_NAMES = ["commonsense", "deontology", "justice", "virtue", "utilitarian"]


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SingleTaskEthicsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, task_name: str, max_length: int = 128):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        text = item["text"]
        label = item["labels"][self.task_name]  # 0/1

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(label, dtype=torch.long)
        return encoded


def compute_metrics(y_true: List[int], y_prob: List[float]):
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def train_single_task(task_name: str,
                      train_file: str,
                      val_file: str,
                      output_dir: str,
                      epochs: int = 2,
                      lr: float = 2e-5):

    print(f"\n===== Training single-task BERT for [{task_name}] =====")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    dataset = load_dataset("json", data_files={
        "train": train_file,
        "validation": val_file,
    })

    train_ds = SingleTaskEthicsDataset(dataset["train"], tokenizer, task_name)
    val_ds = SingleTaskEthicsDataset(dataset["validation"], tokenizer, task_name)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[{task_name}] Epoch {epoch} - Train loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].numpy().tolist()
                y_true.extend(labels)

                inputs = {
                    "input_ids": batch["input_ids"].to(DEVICE),
                    "attention_mask": batch["attention_mask"].to(DEVICE),
                }
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                y_prob.extend(probs.cpu().numpy().tolist())

        metrics = compute_metrics(y_true, y_prob)
        print(
            f"[{task_name}] Val - F1={metrics['f1']:.3f}, "
            f"ACC={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_path = os.path.join(output_dir, f"best_{task_name}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[{task_name}] Saved best model to {save_path} (F1={best_f1:.3f})")


def main():
    set_seed(42)
    print(f"Using device: {DEVICE}")

    train_file = "data/ethics_multitask_train.jsonl"
    val_file = "data/ethics_multitask_val.jsonl"
    out_dir = "models/single_task_bert"

    for task in TASK_NAMES:
        train_single_task(
            task_name=task,
            train_file=train_file,
            val_file=val_file,
            output_dir=out_dir,
            epochs=2,
            lr=2e-5,
        )


if __name__ == "__main__":
    main()
