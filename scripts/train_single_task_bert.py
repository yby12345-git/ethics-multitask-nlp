"""
train_single_task_bert_all_tasks.py

Purpose
-------
Train an independent BERT classifier for each ethical reasoning task in the
multi-task ETHICS dataset.

The script performs the following steps:

1. Load the multi-task training and validation JSONL files.
2. Validate the required text and label fields.
3. Create one task-specific dataset for each ethical reasoning task.
4. Fine-tune an independent BERT classifier for each task.
5. Evaluate each model after every training epoch.
6. Save the best-performing checkpoint according to validation F1-score.
7. Export the best validation metrics for all tasks.

Expected JSONL structure
------------------------
Each record must contain:

{
    "text": "Example sentence",
    "labels": {
        "commonsense": 0,
        "deontology": 1,
        "justice": 0,
        "virtue": 1,
        "utilitarian": 0
    }
}

Generated outputs
-----------------
models/single_task_bert/
    commonsense/
    deontology/
    justice/
    virtue/
    utilitarian/
    best_metrics.json

Required packages
-----------------
torch
transformers
datasets
scikit-learn
numpy
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)


# ===================== Global configuration =====================

MODEL_NAME = "bert-base-uncased"

TASK_NAMES: Tuple[str, ...] = (
    "commonsense",
    "deontology",
    "justice",
    "virtue",
    "utilitarian",
)

RANDOM_SEED = 42
MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "ethics_multitask_train.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "ethics_multitask_val.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "single_task_bert"
)

METRICS_OUTPUT_PATH = (
    OUTPUT_DIR
    / "best_metrics.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ===================== Reproducibility =====================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Configure random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed used by Python, NumPy, PyTorch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Initialize a deterministic random seed for a DataLoader worker.

    Parameters
    ----------
    worker_id : int
        Worker identifier assigned by PyTorch.
    """
    worker_seed = (
        torch.initial_seed() + worker_id
    ) % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===================== Dataset implementation =====================

class SingleTaskEthicsDataset(Dataset):
    """
    PyTorch dataset for one ethical classification task.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        Hugging Face dataset containing ``text`` and ``labels`` fields.

    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer associated with the pretrained language model.

    task_name : str
        Ethical task whose binary label should be used.

    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        hf_dataset: HuggingFaceDataset,
        tokenizer: PreTrainedTokenizerBase,
        task_name: str,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if task_name not in TASK_NAMES:
            raise ValueError(
                f"Unsupported task name '{task_name}'. "
                f"Expected one of: {list(TASK_NAMES)}"
            )

        if max_length < 1:
            raise ValueError(
                "max_length must be at least 1."
            )

        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        """
        Return one tokenized example and its binary task label.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict
            Tokenized model inputs and a ``labels`` value.
        """
        item = self.dataset[index]

        text = str(item["text"]).strip()
        label_mapping = item["labels"]

        if not text:
            raise ValueError(
                f"Sample {index} contains empty text."
            )

        if not isinstance(label_mapping, Mapping):
            raise TypeError(
                f"Sample {index} has an invalid 'labels' field."
            )

        if self.task_name not in label_mapping:
            raise KeyError(
                f"Sample {index} does not contain a label "
                f"for task '{self.task_name}'."
            )

        label = int(label_mapping[self.task_name])

        if label not in {0, 1}:
            raise ValueError(
                f"Sample {index} contains invalid label {label} "
                f"for task '{self.task_name}'."
            )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        encoded["labels"] = label

        return encoded


# ===================== Dataset loading and validation =====================

def validate_input_files(
    train_file: Path,
    validation_file: Path,
) -> None:
    """
    Confirm that the required JSONL files exist.

    Parameters
    ----------
    train_file : pathlib.Path
        Training JSONL path.

    validation_file : pathlib.Path
        Validation JSONL path.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    """
    missing_files = [
        path
        for path in (
            train_file,
            validation_file,
        )
        if not path.exists()
    ]

    if missing_files:
        formatted_paths = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following input files were not found:\n"
            f"{formatted_paths}"
        )


def load_multitask_dataset(
    train_file: Path,
    validation_file: Path,
) -> DatasetDict:
    """
    Load training and validation data from JSONL files.

    Parameters
    ----------
    train_file : pathlib.Path
        Path to the training JSONL file.

    validation_file : pathlib.Path
        Path to the validation JSONL file.

    Returns
    -------
    datasets.DatasetDict
        Dataset containing training and validation splits.
    """
    validate_input_files(
        train_file,
        validation_file,
    )

    print("Loading multi-task ETHICS datasets.")
    print(f"Training file: {train_file}")
    print(f"Validation file: {validation_file}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(validation_file),
        },
    )

    validate_dataset_structure(dataset)

    print("Dataset sizes:")
    print(f"  Training: {len(dataset['train']):,}")
    print(
        f"  Validation: "
        f"{len(dataset['validation']):,}"
    )

    return dataset


def validate_dataset_structure(
    dataset: DatasetDict,
) -> None:
    """
    Validate dataset splits, columns, tasks, and binary labels.

    Parameters
    ----------
    dataset : datasets.DatasetDict
        Multi-task dataset to validate.

    Raises
    ------
    ValueError
        If required splits, columns, labels, or samples are missing.
    """
    required_splits = {
        "train",
        "validation",
    }

    missing_splits = (
        required_splits
        - set(dataset.keys())
    )

    if missing_splits:
        raise ValueError(
            "The dataset is missing required splits: "
            f"{sorted(missing_splits)}"
        )

    required_columns = {
        "text",
        "labels",
    }

    for split_name in required_splits:
        split = dataset[split_name]

        if len(split) == 0:
            raise ValueError(
                f"The '{split_name}' split is empty."
            )

        missing_columns = (
            required_columns
            - set(split.column_names)
        )

        if missing_columns:
            raise ValueError(
                f"The '{split_name}' split is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for sample_index, sample in enumerate(split):
            text = str(sample["text"]).strip()
            labels = sample["labels"]

            if not text:
                raise ValueError(
                    f"Empty text found in split '{split_name}' "
                    f"at sample {sample_index}."
                )

            if not isinstance(labels, Mapping):
                raise TypeError(
                    f"Invalid labels object in split '{split_name}' "
                    f"at sample {sample_index}."
                )

            missing_tasks = (
                set(TASK_NAMES)
                - set(labels.keys())
            )

            if missing_tasks:
                raise ValueError(
                    f"Sample {sample_index} in split "
                    f"'{split_name}' is missing task labels: "
                    f"{sorted(missing_tasks)}"
                )

            for task_name in TASK_NAMES:
                label = int(labels[task_name])

                if label not in {0, 1}:
                    raise ValueError(
                        f"Invalid label {label} for task "
                        f"'{task_name}' in split '{split_name}' "
                        f"at sample {sample_index}."
                    )


def print_task_distributions(
    dataset: DatasetDict,
) -> None:
    """
    Print class distributions for every task and split.

    Parameters
    ----------
    dataset : datasets.DatasetDict
        Validated multi-task dataset.
    """
    print("\nTask label distributions:")

    for split_name in (
        "train",
        "validation",
    ):
        print(f"\n  {split_name.capitalize()} split:")

        for task_name in TASK_NAMES:
            labels = [
                int(sample["labels"][task_name])
                for sample in dataset[split_name]
            ]

            negative_count = labels.count(0)
            positive_count = labels.count(1)

            print(
                f"    {task_name}: "
                f"label 0 = {negative_count:,}, "
                f"label 1 = {positive_count:,}"
            )


# ===================== Evaluation metrics =====================

def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> Dict[str, float]:
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    y_true : sequence of int
        Ground-truth binary labels.

    y_prob : sequence of float
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score, and AUC.

    Raises
    ------
    ValueError
        If the input sequences have different lengths or are empty.
    """
    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must contain the same "
            "number of values."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Metric calculation requires at least one sample."
        )

    y_true_array = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob_array = np.asarray(
        y_prob,
        dtype=float,
    )

    y_pred_array = (
        y_prob_array >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true_array,
        y_pred_array,
    )

    precision = precision_score(
        y_true_array,
        y_pred_array,
        zero_division=0,
    )

    recall = recall_score(
        y_true_array,
        y_pred_array,
        zero_division=0,
    )

    f1 = f1_score(
        y_true_array,
        y_pred_array,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            y_true_array,
            y_prob_array,
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


# ===================== Optimizer configuration =====================

def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """
    Create an AdamW optimizer with separate decay parameter groups.

    Bias parameters and layer-normalization parameters are excluded from
    weight decay.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters should be optimized.

    learning_rate : float
        Initial learning rate.

    weight_decay : float
        Weight decay applied to eligible parameters.

    Returns
    -------
    torch.optim.AdamW
        Configured optimizer.
    """
    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )

    optimizer_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and not any(
                    term in name
                    for term in no_decay_terms
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and any(
                    term in name
                    for term in no_decay_terms
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        optimizer_groups,
        lr=learning_rate,
    )


# ===================== Training and validation =====================

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    """
    Train a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Classification model.

    data_loader : torch.utils.data.DataLoader
        Training DataLoader.

    optimizer : torch.optim.AdamW
        Model optimizer.

    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning-rate scheduler.

    Returns
    -------
    float
        Average training loss.
    """
    if len(data_loader) == 0:
        raise ValueError(
            "The training DataLoader contains no batches."
        )

    model.train()
    total_loss = 0.0

    for batch in data_loader:
        optimizer.zero_grad(
            set_to_none=True
        )

        batch = {
            key: value.to(
                DEVICE,
                non_blocking=True,
            )
            for key, value in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss

        if loss is None:
            raise RuntimeError(
                "The model did not return a training loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()
        scheduler.step()

        total_loss += float(
            loss.detach().item()
        )

    return total_loss / len(data_loader)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate a model on a validation dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Classification model.

    data_loader : torch.utils.data.DataLoader
        Validation DataLoader.

    Returns
    -------
    dict
        Validation loss and classification metrics.
    """
    if len(data_loader) == 0:
        raise ValueError(
            "The validation DataLoader contains no batches."
        )

    model.eval()

    total_loss = 0.0
    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                key: value.to(
                    DEVICE,
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            outputs = model(**batch)

            if outputs.loss is not None:
                total_loss += float(
                    outputs.loss.detach().item()
                )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[:, 1]

            y_true.extend(
                batch["labels"]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

            y_prob.extend(
                probabilities
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )

    metrics = compute_metrics(
        y_true,
        y_prob,
    )

    metrics["loss"] = (
        total_loss / len(data_loader)
    )

    return metrics


# ===================== Output utilities =====================

def save_json(
    data: Mapping[str, Any],
    output_path: Path,
) -> None:
    """
    Save a mapping as a UTF-8 encoded JSON file.

    Parameters
    ----------
    data : mapping
        Data to serialize.

    output_path : pathlib.Path
        Destination JSON path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_data = {}

    for key, value in data.items():
        if isinstance(
            value,
            (np.integer, np.floating),
        ):
            serializable_data[key] = value.item()
        else:
            serializable_data[key] = value

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_best_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_path: Path,
    metrics: Mapping[str, float],
) -> None:
    """
    Save the best model, tokenizer, and validation metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Best-performing model.

    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer associated with the model.

    output_path : pathlib.Path
        Task-specific output directory.

    metrics : mapping
        Validation metrics associated with the checkpoint.
    """
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_path
    )

    tokenizer.save_pretrained(
        output_path
    )

    save_json(
        metrics,
        output_path / "validation_metrics.json",
    )


# ===================== Task-specific workflow =====================

def train_single_task(
    task_name: str,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> Dict[str, float]:
    """
    Train and validate one independent BERT classifier.

    Parameters
    ----------
    task_name : str
        Ethical classification task.

    dataset : datasets.DatasetDict
        Multi-task training and validation data.

    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer associated with the pretrained model.

    output_dir : pathlib.Path
        Parent model output directory.

    epochs : int
        Number of training epochs.

    learning_rate : float
        Initial optimizer learning rate.

    Returns
    -------
    dict
        Best validation metrics for the task.
    """
    if task_name not in TASK_NAMES:
        raise ValueError(
            f"Unsupported task name: {task_name}"
        )

    if epochs < 1:
        raise ValueError(
            "epochs must be at least 1."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    print(
        "\n=================================================="
    )
    print(
        f"Training single-task BERT for: {task_name}"
    )
    print(
        "=================================================="
    )

    train_dataset = SingleTaskEthicsDataset(
        dataset["train"],
        tokenizer,
        task_name,
        MAX_LENGTH,
    )

    validation_dataset = SingleTaskEthicsDataset(
        dataset["validation"],
        tokenizer,
        task_name,
        MAX_LENGTH,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    generator = torch.Generator()
    generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={
                0: "negative",
                1: "positive",
            },
            label2id={
                "negative": 0,
                "positive": 1,
            },
        )
    )

    model.to(DEVICE)

    optimizer = create_optimizer(
        model,
        learning_rate,
        WEIGHT_DECAY,
    )

    total_training_steps = (
        len(train_loader) * epochs
    )

    if total_training_steps < 1:
        raise ValueError(
            "The calculated number of training steps is zero."
        )

    warmup_steps = int(
        WARMUP_RATIO
        * total_training_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_f1 = -math.inf
    best_metrics: Dict[str, float] = {}

    task_output_dir = (
        output_dir
        / task_name
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        average_training_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"[{task_name}] Epoch "
            f"{epoch}/{epochs}"
        )

        print(
            f"  Training loss: "
            f"{average_training_loss:.6f}"
        )

        print(
            f"  Validation loss: "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"  Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"  F1-score: "
            f"{validation_metrics['f1']:.4f}"
        )

        print(
            f"  AUC: "
            f"{validation_metrics['auc']:.4f}"
        )

        print(
            f"  Learning rate: "
            f"{current_learning_rate:.8f}"
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]

            best_metrics = {
                "task": task_name,
                "epoch": epoch,
                "training_loss": (
                    average_training_loss
                ),
                **validation_metrics,
            }

            save_best_model(
                model,
                tokenizer,
                task_output_dir,
                best_metrics,
            )

            print(
                f"  Saved new best model to: "
                f"{task_output_dir}"
            )

    if not best_metrics:
        raise RuntimeError(
            f"No model checkpoint was saved for task "
            f"'{task_name}'."
        )

    del model
    del optimizer
    del scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_metrics


# ===================== Main workflow =====================

def main() -> None:
    """
    Train independent BERT classifiers for all ethical tasks.
    """
    set_seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print(
        "Starting single-task BERT experiments."
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device_name}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Tasks: {list(TASK_NAMES)}")

    dataset = load_multitask_dataset(
        TRAIN_FILE,
        VALIDATION_FILE,
    )

    print_task_distributions(
        dataset
    )

    print(
        f"\nLoading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    all_best_metrics: Dict[
        str,
        Dict[str, float],
    ] = {}

    for task_name in TASK_NAMES:
        set_seed(
            RANDOM_SEED
        )

        best_metrics = train_single_task(
            task_name=task_name,
            dataset=dataset,
            tokenizer=tokenizer,
            output_dir=OUTPUT_DIR,
            epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
        )

        all_best_metrics[task_name] = (
            best_metrics
        )

    save_json(
        all_best_metrics,
        METRICS_OUTPUT_PATH,
    )

    print(
        "\n=================================================="
    )
    print(
        "All single-task experiments completed successfully."
    )
    print(
        "=================================================="
    )

    print(
        f"Model directory: {OUTPUT_DIR}"
    )

    print(
        f"Metrics file: {METRICS_OUTPUT_PATH}"
    )

    print("\nBest validation results:")

    for task_name, metrics in all_best_metrics.items():
        print(
            f"  {task_name}: "
            f"F1={metrics['f1']:.4f}, "
            f"ACC={metrics['accuracy']:.4f}, "
            f"AUC={metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
