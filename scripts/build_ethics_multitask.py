"""
build_ethics_multitask.py

Purpose
-------
Convert the raw ETHICS benchmark CSV files into unified JSON Lines files
for multi-task ethical classification.

The script processes the following ETHICS subtasks:
- Commonsense
- Deontology
- Justice
- Virtue
- Utilitarianism

It creates reproducible training, validation, and test datasets.
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional


# ===================== Configuration =====================

RANDOM_SEED = 42
VALIDATION_RATIO = 0.10
LABEL_DETECTION_SAMPLE_SIZE = 200

TASKS = {
    "commonsense": "commonsense",
    "deontology": "deontology",
    "justice": "justice",
    "virtue": "virtue",
    "utilitarian": "utilitarianism",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_ROOT = PROJECT_ROOT / "data" / "ethics_raw"

TRAIN_OUTPUT_PATH = PROJECT_ROOT / "data" / "ethics_multitask_train.jsonl"
VALIDATION_OUTPUT_PATH = PROJECT_ROOT / "data" / "ethics_multitask_val.jsonl"
TEST_OUTPUT_PATH = PROJECT_ROOT / "data" / "ethics_multitask_test.jsonl"

EXCLUDED_UTILITARIAN_FIELDS = {
    "label",
    "gold",
    "gold_label",
    "id",
    "group_id",
    "index",
    "less_pleasant",
}


# ===================== Text construction =====================

def build_text(task_name: str, row: Dict[str, str]) -> str:
    """
    Construct an input text string from one CSV row.

    Parameters
    ----------
    task_name : str
        Internal name of the ETHICS subtask.

    row : dict
        CSV row containing the original dataset fields.

    Returns
    -------
    str
        Processed input text.

    Raises
    ------
    ValueError
        If the task name is unsupported.
    """
    if task_name == "commonsense":
        text = row.get("input", "")

    elif task_name == "deontology":
        scenario = row.get("scenario", "")
        excuse = row.get("excuse", "")
        text = f"{scenario} {excuse}"

    elif task_name in {"justice", "virtue"}:
        text = row.get("scenario", "")

    elif task_name == "utilitarian":
        parts = []

        for key, value in row.items():
            normalized_key = key.lower().strip()

            if normalized_key in EXCLUDED_UTILITARIAN_FIELDS:
                continue

            normalized_value = str(value).strip()

            if normalized_value:
                parts.append(normalized_value)

        text = " ".join(parts)

    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    return str(text).strip()


# ===================== CSV utilities =====================

def load_csv(path: Path) -> Iterator[Dict[str, str]]:
    """
    Yield rows from a UTF-8 encoded CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.

    Yields
    ------
    dict
        One CSV row at a time.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"The CSV file does not contain a header: {path}")

        for row in reader:
            yield row


def guess_label_key(path: Path) -> str:
    """
    Identify the binary label column in a CSV file.

    Columns containing the word ``label`` are preferred. If no such
    column is available, the function searches for a column whose
    observed values are exclusively 0 and 1.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.

    Returns
    -------
    str
        Name of the detected label column.

    Raises
    ------
    ValueError
        If the file is empty or no binary label column can be found.
    """
    sampled_rows: List[Dict[str, str]] = []

    for index, row in enumerate(load_csv(path)):
        sampled_rows.append(row)

        if index + 1 >= LABEL_DETECTION_SAMPLE_SIZE:
            break

    if not sampled_rows:
        raise ValueError(f"No data rows were found in: {path}")

    column_names = list(sampled_rows[0].keys())

    for key in column_names:
        if "label" in key.lower():
            print(
                f"  > Selected label-like column in "
                f"{path.name}: {key}"
            )
            return key

    for key in column_names:
        observed_values = {
            str(row.get(key, "")).strip()
            for row in sampled_rows
            if str(row.get(key, "")).strip()
        }

        if observed_values and observed_values.issubset({"0", "1"}):
            print(
                f"  > Detected binary label column in "
                f"{path.name}: {key}"
            )
            return key

    raise ValueError(
        f"No binary label column could be identified in {path}. "
        f"Available columns: {column_names}"
    )


# ===================== Sample construction =====================

def create_empty_labels() -> Dict[str, int]:
    """
    Create the default label dictionary for all subtasks.

    Returns
    -------
    dict
        Dictionary initialized with zero for each task.
    """
    return {task_name: 0 for task_name in TASKS}


def create_entry(
    task_name: str,
    text: str,
    label: int,
) -> Dict[str, object]:
    """
    Create one multi-task dataset entry.

    Parameters
    ----------
    task_name : str
        Name of the active subtask.

    text : str
        Processed input text.

    label : int
        Target label for the active subtask.

    Returns
    -------
    dict
        JSON-serializable dataset entry.
    """
    labels = create_empty_labels()
    labels[task_name] = int(label)

    return {
        "text": text,
        "task": task_name,
        "labels": labels,
    }


def parse_binary_label(
    row: Dict[str, str],
    label_key: str,
    path: Path,
) -> int:
    """
    Parse and validate a binary class label.

    Parameters
    ----------
    row : dict
        Original CSV row.

    label_key : str
        Name of the label column.

    path : pathlib.Path
        Source CSV file, used in error messages.

    Returns
    -------
    int
        Binary class label.

    Raises
    ------
    ValueError
        If the label is missing or is not 0 or 1.
    """
    raw_label = str(row.get(label_key, "")).strip()

    if raw_label not in {"0", "1"}:
        raise ValueError(
            f"Invalid binary label {raw_label!r} in {path}. "
            "Expected either '0' or '1'."
        )

    return int(raw_label)


def process_binary_task(
    task_name: str,
    train_path: Path,
    test_path: Path,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Process one binary ETHICS subtask.

    Returns
    -------
    tuple
        Training entries and test entries for the selected subtask.
    """
    train_entries: List[Dict[str, object]] = []
    test_entries: List[Dict[str, object]] = []

    train_label_key = guess_label_key(train_path)
    test_label_key = guess_label_key(test_path)

    for row in load_csv(train_path):
        text = build_text(task_name, row)

        if not text:
            continue

        label = parse_binary_label(row, train_label_key, train_path)
        train_entries.append(create_entry(task_name, text, label))

    for row in load_csv(test_path):
        text = build_text(task_name, row)

        if not text:
            continue

        label = parse_binary_label(row, test_label_key, test_path)
        test_entries.append(create_entry(task_name, text, label))

    return train_entries, test_entries


def process_utilitarian_task(
    task_name: str,
    train_path: Path,
    test_path: Path,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Process the ETHICS Utilitarianism subtask.

    The categorical label mapping is created exclusively from the
    training set and is then applied unchanged to the test set.

    Returns
    -------
    tuple
        Training entries and test entries for the utilitarianism task.

    Raises
    ------
    ValueError
        If a label is missing or an unseen test label is encountered.
    """
    train_entries: List[Dict[str, object]] = []
    test_entries: List[Dict[str, object]] = []
    label_map: Dict[str, int] = {}

    print(
        "  > Utilitarianism: using 'less_pleasant' "
        "as the categorical label."
    )

    for row in load_csv(train_path):
        text = build_text(task_name, row)
        raw_label = str(row.get("less_pleasant", "")).strip()

        if not text:
            continue

        if not raw_label:
            raise ValueError(
                f"Missing 'less_pleasant' label in {train_path}"
            )

        if raw_label not in label_map:
            label_map[raw_label] = len(label_map)

        train_entries.append(
            create_entry(
                task_name,
                text,
                label_map[raw_label],
            )
        )

    if not label_map:
        raise ValueError(
            "No utilitarianism labels were found in the training set."
        )

    for row in load_csv(test_path):
        text = build_text(task_name, row)
        raw_label = str(row.get("less_pleasant", "")).strip()

        if not text:
            continue

        if not raw_label:
            raise ValueError(
                f"Missing 'less_pleasant' label in {test_path}"
            )

        if raw_label not in label_map:
            raise ValueError(
                f"Unseen utilitarianism label {raw_label!r} "
                f"encountered in the test set."
            )

        test_entries.append(
            create_entry(
                task_name,
                text,
                label_map[raw_label],
            )
        )

    print(f"  > Utilitarianism label mapping: {label_map}")

    return train_entries, test_entries


# ===================== Output utilities =====================

def write_jsonl(
    path: Path,
    data: List[Dict[str, object]],
) -> None:
    """
    Write dataset entries to a UTF-8 encoded JSON Lines file.

    Parameters
    ----------
    path : pathlib.Path
        Destination JSONL file.

    data : list
        Dataset entries to be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for entry in data:
            file.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ===================== Main workflow =====================

def main() -> None:
    """Build reproducible multi-task ETHICS datasets."""
    random.seed(RANDOM_SEED)

    train_data: List[Dict[str, object]] = []
    test_data: List[Dict[str, object]] = []

    for task_name, subset_name in TASKS.items():
        subset_dir = RAW_ROOT / subset_name
        train_path = subset_dir / "train.csv"
        test_path = subset_dir / "test.csv"

        print(
            f"\nProcessing subset [{subset_name}] "
            f"for task [{task_name}]..."
        )

        if task_name == "utilitarian":
            task_train, task_test = process_utilitarian_task(
                task_name,
                train_path,
                test_path,
            )
        else:
            task_train, task_test = process_binary_task(
                task_name,
                train_path,
                test_path,
            )

        train_data.extend(task_train)
        test_data.extend(task_test)

        print(f"  > Training samples: {len(task_train):,}")
        print(f"  > Test samples: {len(task_test):,}")

    print(
        f"\nTotal training samples before validation split: "
        f"{len(train_data):,}"
    )
    print(f"Total test samples: {len(test_data):,}")

    if not train_data:
        raise ValueError("No training samples were generated.")

    if not test_data:
        raise ValueError("No test samples were generated.")

    random.shuffle(train_data)

    validation_size = max(
        1,
        int(VALIDATION_RATIO * len(train_data)),
    )

    validation_data = train_data[:validation_size]
    final_train_data = train_data[validation_size:]

    write_jsonl(TRAIN_OUTPUT_PATH, final_train_data)
    write_jsonl(VALIDATION_OUTPUT_PATH, validation_data)
    write_jsonl(TEST_OUTPUT_PATH, test_data)

    print("\nDataset construction completed successfully.")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Validation ratio: {VALIDATION_RATIO:.0%}")
    print(f"Training samples: {len(final_train_data):,}")
    print(f"Validation samples: {len(validation_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print("\nGenerated files:")
    print(f"  - {TRAIN_OUTPUT_PATH}")
    print(f"  - {VALIDATION_OUTPUT_PATH}")
    print(f"  - {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
