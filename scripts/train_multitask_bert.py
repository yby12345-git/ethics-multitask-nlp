"""
train_single_task_bert.py

Purpose
-------
Fine-tune BERT for single-task ethical classification using the
Commonsense subset of the ETHICS benchmark.

The script performs the following steps:

1. Load the ETHICS Commonsense dataset.
2. Create reproducible training and validation splits.
3. Tokenize the input text.
4. Fine-tune a pretrained BERT classification model.
5. select the best checkpoint according to validation F1-score.
6. Evaluate the selected model on the held-out test set.
7. Save the model, tokenizer, and evaluation metrics.

Required packages
-----------------
torch
transformers
datasets
scikit-learn
numpy
"""

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ===================== Configuration =====================

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "ethics"
DATASET_CONFIG = "commonsense"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.10
MAX_LENGTH = 128

LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "single_task_bert"
LOGGING_DIR = PROJECT_ROOT / "logs" / "single_task_bert"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
METRICS_OUTPUT_PATH = OUTPUT_DIR / "test_metrics.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGGING_DIR.mkdir(parents=True, exist_ok=True)


# ===================== Reproducibility =====================

def configure_reproducibility(seed: int) -> None:
    """
    Configure random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed used by Python, NumPy, PyTorch, and Transformers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===================== Dataset preparation =====================

def load_and_split_dataset() -> DatasetDict:
    """
    Load the ETHICS Commonsense dataset and create a validation split.

    The original training set is divided into training and validation
    subsets using a reproducible stratified split. The original test set
    remains untouched until final evaluation.

    Returns
    -------
    datasets.DatasetDict
        Dataset dictionary containing train, validation, and test splits.

    Raises
    ------
    ValueError
        If required dataset columns or splits are missing.
    """
    print(
        f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}"
    )

    raw_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    required_splits = {
        "train",
        "test",
    }

    missing_splits = (
        required_splits
        - set(raw_dataset.keys())
    )

    if missing_splits:
        raise ValueError(
            "The dataset is missing the following required splits: "
            f"{sorted(missing_splits)}"
        )

    required_columns = {
        "input",
        "label",
    }

    missing_columns = (
        required_columns
        - set(raw_dataset["train"].column_names)
    )

    if missing_columns:
        raise ValueError(
            "The training split is missing the following columns: "
            f"{sorted(missing_columns)}"
        )

    split_dataset = raw_dataset["train"].train_test_split(
        test_size=VALIDATION_RATIO,
        seed=RANDOM_SEED,
        shuffle=True,
        stratify_by_column="label",
    )

    dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
            "test": raw_dataset["test"],
        }
    )

    print("Dataset split sizes:")
    print(f"  Training: {len(dataset['train']):,}")
    print(f"  Validation: {len(dataset['validation']):,}")
    print(f"  Test: {len(dataset['test']):,}")

    return dataset


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
) -> DatasetDict:
    """
    Tokenize all dataset splits.

    Parameters
    ----------
    dataset : datasets.DatasetDict
        Dataset containing train, validation, and test splits.

    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer associated with the pretrained model.

    Returns
    -------
    datasets.DatasetDict
        Tokenized dataset.
    """

    def preprocess_batch(
        examples: Dict[str, list],
    ) -> Dict[str, list]:
        """
        Tokenize a batch of input examples.
        """
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print(
        f"Tokenizing dataset with maximum length {MAX_LENGTH}."
    )

    tokenized_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        desc="Tokenizing ETHICS Commonsense dataset",
    )

    columns_to_remove = [
        column
        for column in tokenized_dataset["train"].column_names
        if column not in {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "label",
        }
    ]

    tokenized_dataset = tokenized_dataset.remove_columns(
        columns_to_remove
    )

    return tokenized_dataset


# ===================== Evaluation metrics =====================

def compute_metrics(
    evaluation_prediction: EvalPrediction,
) -> Dict[str, float]:
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    evaluation_prediction : transformers.EvalPrediction
        Model predictions and reference labels.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score, and AUC.
    """
    logits = evaluation_prediction.predictions
    labels = evaluation_prediction.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_labels = np.argmax(
        logits,
        axis=-1,
    )

    shifted_logits = (
        logits
        - np.max(
            logits,
            axis=-1,
            keepdims=True,
        )
    )

    probabilities = np.exp(shifted_logits)
    probabilities = (
        probabilities
        / probabilities.sum(
            axis=-1,
            keepdims=True,
        )
    )

    positive_class_probabilities = probabilities[:, 1]

    accuracy = accuracy_score(
        labels,
        predicted_labels,
    )

    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="binary",
            zero_division=0,
        )
    )

    try:
        auc = roc_auc_score(
            labels,
            positive_class_probabilities,
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "auc": float(auc),
    }


# ===================== Output utilities =====================

def save_metrics(
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """
    Save evaluation metrics as a JSON file.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics returned by the Trainer.

    output_path : pathlib.Path
        Destination JSON file.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_metrics = {}

    for key, value in metrics.items():
        if isinstance(
            value,
            (np.integer, np.floating),
        ):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Saved evaluation metrics: {output_path}")


# ===================== Main workflow =====================

def main() -> None:
    """
    Execute single-task BERT training and evaluation.
    """
    configure_reproducibility(
        RANDOM_SEED
    )

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print("Starting single-task BERT training.")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device_name}")
    print(f"Random seed: {RANDOM_SEED}")

    dataset = load_and_split_dataset()

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
    )

    print(
        f"Loading classification model: {MODEL_NAME}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "unethical",
            1: "ethical",
        },
        label2id={
            "unethical": 0,
            "ethical": 1,
        },
    )

    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=str(LOGGING_DIR),
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting model training.")

    training_result = trainer.train()

    trainer.save_state()

    print("\nTraining completed.")
    print(
        f"Training loss: "
        f"{training_result.training_loss:.6f}"
    )

    print("\nEvaluating the best model on the test set.")

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print("\nTest results:")

    for metric_name in sorted(test_metrics):
        metric_value = test_metrics[metric_name]

        if isinstance(metric_value, float):
            print(
                f"  {metric_name}: "
                f"{metric_value:.6f}"
            )
        else:
            print(
                f"  {metric_name}: "
                f"{metric_value}"
            )

    BEST_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(BEST_MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(BEST_MODEL_DIR)
    )

    save_metrics(
        test_metrics,
        METRICS_OUTPUT_PATH,
    )

    print(
        "\nSingle-task BERT experiment "
        "completed successfully."
    )

    print(
        f"Best model directory: {BEST_MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
