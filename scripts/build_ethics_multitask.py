import os
import csv
import json
import random

# 模型中使用的任务名
TASKS = {
    "commonsense": "commonsense",
    "deontology": "deontology",
    "justice": "justice",
    "virtue": "virtue",
    "utilitarian": "utilitarianism",
}

RAW_ROOT = "data/ethics_raw"


def build_text(task_name: str, row: dict) -> str:
    """根据子任务，从 CSV 行里拼出 text 字段。"""
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
        # utilitarian：把除 label/id 外所有字段拼起来
        parts = []
        for k, v in row.items():
            kl = k.lower()
            if kl in ("label", "gold", "gold_label", "id", "group_id", "index", "less_pleasant"):
                # less_pleasant 我们单独当标签用，不进文本
                continue
            if str(v).strip():
                parts.append(str(v).strip())
        return " ".join(parts)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def guess_label_key(path: str) -> str:
    """自动猜测 CSV 里哪一列是标签列（0/1）。"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_rows = []
        for i, row in enumerate(reader):
            first_rows.append(row)
            if i >= 199:
                break

    if not first_rows:
        raise ValueError(f"No data rows in {path}")

    # 优先选列名含 label 的
    for key in first_rows[0].keys():
        if "label" in key.lower():
            print(f"  > Prefer label-like column in {os.path.basename(path)}: {key}")
            return key

    # 再选几乎全是 0/1 的列
    keys = list(first_rows[0].keys())
    for key in keys:
        vals = set(row[key].strip() for row in first_rows if row[key].strip() != "")
        if vals.issubset({"0", "1"}):
            print(f"  > Detected 0/1 label column in {os.path.basename(path)}: {key}")
            return key

    raise ValueError(f"Cannot find label-like column in {path}. Columns: {list(first_rows[0].keys())}")


def load_csv(path: str):
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

        # ------ 非 utilitarian：按 0/1 标签处理 ------
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

        # ------ utilitarian：用 less_pleasant 作“多类别标签” ------
        else:
            print("  > utilitarian: using 'less_pleasant' as categorical label")
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
                # 测试集中如果遇到新取值，也自动扩展
                label = encode_label(raw_label)
                entry = {
                    "text": text,
                    "labels": {t: 0 for t in TASKS},
                }
                entry["labels"][task_name] = label
                test_data.append(entry)

            print(f"  > utilitarian label mapping: {label_map}")

    print(f"Total train samples (before split): {len(train_data)}")
    print(f"Total test  samples: {len(test_data)}")

    random.seed(42)
    random.shuffle(train_data)
    val_size = int(0.1 * len(train_data))
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    def write_jsonl(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_jsonl("data/ethics_multitask_train.jsonl", train_data)
    write_jsonl("data/ethics_multitask_val.jsonl", val_data)
    write_jsonl("data/ethics_multitask_test.jsonl", test_data)

    print("Done! JSONL files saved to data/:")
    print("  - data/ethics_multitask_train.jsonl")
    print("  - data/ethics_multitask_val.jsonl")
    print("  - data/ethics_multitask_test.jsonl")


if __name__ == "__main__":
    main()
