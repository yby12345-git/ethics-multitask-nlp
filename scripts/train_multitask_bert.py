import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ======================
# 1 Load dataset
# ======================
dataset = load_dataset("ethics", "commonsense")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(preprocess)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ======================
# 2 Load model
# ======================
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    use_safetensors=False,   # 关键：关闭 safetensors
    force_download=True      # 关键：强制重新下载
)

# ======================
# 3 Training args
# ======================
training_args = TrainingArguments(
    output_dir="../outputs",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
    save_strategy="epoch"
)

# ======================
# 4 Trainer
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# ======================
# 5 Train
# ======================
trainer.train()

# ======================
# 6 Evaluate
# ======================
metrics = trainer.evaluate()
print(metrics)
