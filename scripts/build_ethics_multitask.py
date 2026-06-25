import os
import csv
import json
import random

# Task names used in the model
TASKS = {
    "commonsense": "commonsense",
    "deontology": "deontology",
    "justice": "justice",
    "virtue": "virtue",
    "utilitarian": "utilitarianism",
}

RAW_ROOT = "data/ethics_raw"


def build_text(task_name: str, row: dict) -> str:
    """Construct the text field from a CSV row according to the subtask."""
    if task_name == "commonsense":
        return row.get("input", "")

    elif task_name == "deontology":
        scenario = row.get("scenario", "")
        excuse = row.get("excuse", "")
        return f"{scenario} {excuse}".strip()

    elif task_name == "justice":
        return row.get("scenario", "")

    elif task_name == "virtue":
        return row.get("scenario", "")

    elif task_name == "utilitarian":
        # For the utilitarianism task, concatenate all fields except labels and IDs.
        parts = []
        for k, v in row.items():
            kl = k.lower()
            if kl in (
                "label",
                "gold",
                "gold_label",
                "id",
                "group_id",
                "index",
                "less_pleasant",
            ):
                # The less_pleasant field is used separately as the label
                # and is therefore excluded from the input text.
                continue

            if str(v).strip():
                parts.append(str(v).strip())

        return " ".join(parts)

    else:
        raise ValueError(f"Unknown task: {task_name}")


def guess_label_key(path: str) -> str:
    """Automatically identify the binary label column in a CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_rows = []

        for i, row in enumerate(reader):
            first_rows.append(row)
            if i >= 199:
                break

    if not first_rows:
        raise ValueError(f"No data rows in {path}")

    # Prefer columns whose names contain the word "label".
    for key in first_rows[0].keys():
        if "label" in key.lower():
            print(f"  > Prefer label-like column in {os.path.basename(path)}: {key}")
            return key

    # Otherwise, select a column whose values are almost entirely binary.
    keys = list(first_rows[0].keys())
    for key in keys:
        vals = set(row[key].strip() for row in first_rows if row[key].strip() != "")
        if vals.issubset({"0", "1"}):
            print(f"  > Detected binary label column in {os.path.basename(path)}: {key}")
            return key

    raise ValueError(
        f"Cannot find label-like column in {path}. "
        f"Columns: {list(first_rows[0].keys())}"
    )


def load_csv(path: str):
    """Load rows from a CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main():
    train_data = []
    test_data = []

    for task_name, subset in TASKS.items():
        subset_dir = os.path.join(RAW_ROOT, subset)
        train_path = os.path.join(subset_dir, "train.csv")
        test_path = os.path.join(subset_dir, "test.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing file: {train_path}")

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing file: {test_path}")

        print(f"Processing subset [{subset}] for task [{task_name}] ...")

        # Non-utilitarian tasks are treated as binary classification tasks.
        if task_name != "utilitarian":
            train_label_key = guess_label_key(train_path)
            test_label_key = guess_label_key(test_path)

            for row in load_csv(train_path):
                text = build_text(task_name, row)
                label = int(row[train_label_key])

                entry = {
                    "text": text,
                    "labels": {t: 0 for t in TASKS},
                }

                entry["labels"][task_name] = label
                train_data.append(entry)

            for row in load_csv(test_path):
                text = build_text(task_name, row)
                label = int(row[test_label_key])

                entry = {
                    "text": text,
                    "labels": {t: 0 for t in TASKS},
                }

                entry["labels"][task_name] = label
                test_data.append(entry)

        # The utilitarianism task uses less_pleasant as a categorical label.
        else:
            print("  > utilitarian: using 'less_pleasant' as the categorical label")
            label_map = {}

            def encode_label(val: str) -> int:
                val = val.strip()
                if val not in label_map:
                    label_map[val] = len(label_map)
                return label_map[val]

            for row in load_csv(train_path):
                text = build_text(task_name, row)
                raw_label = row.get("less_pleasant", "")
                label = encode_label(raw_label)

                entry = {
                    "text": text,
                    "labels": {t: 0 for t in TASKS},
                }

                entry["labels"][task_name] = label
                train_data.append(entry)

            for row in load_csv(test_path):
                text = build_text(task_name, row)
                raw_label = row.get("less_pleasant", "")

                # If an unseen label appears in the test set,
                # the label mapping is automatically extended.
                label = encode_label(raw_label)

                entry = {
                    "text": text,
                    "labels": {t: 0 for t in TASKS},
                }

                entry["labels"][task_name] = label
                test_data.append(entry)

            print(f"  > utilitarian label mapping: {label_map}")

    print(f"Total train samples before split: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")

    random.seed(42)
    random.shuffle(train_data)

    val_size = int(0.1 * len(train_data))
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    print(
        f"Train: {len(train_data)}, "
        f"Validation: {len(val_data)}, "
        f"Test: {len(test_data)}"
    )

    def write_jsonl(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_jsonl("data/ethics_multitask_train.jsonl", train_data)
    write_jsonl("data/ethics_multitask_val.jsonl", val_data)
    write_jsonl("data/ethics_multitask_test.jsonl", test_data)

    print("Done. JSONL files were saved to the data directory:")
    print("  - data/ethics_multitask_train.jsonl")
    print("  - data/ethics_multitask_val.jsonl")
    print("  - data/ethics_multitask_test.jsonl")


if __name__ == "__main__":
    main()
