"""
prepare_train_val_split.py

Purpose
-------
Prepare training and validation datasets for single-task ethical
classification.

The script reads the raw Commonsense ETHICS training file, validates the
required columns, cleans invalid samples, performs a reproducible stratified
split, and saves the processed datasets as CSV files.

Input
-----
data/ethics_raw/commonsense/train.csv

Outputs
-------
data/ethics_dataset_train.csv
data/ethics_dataset_val.csv
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# ===================== Configuration =====================

RANDOM_SEED = 42
VALIDATION_RATIO = 0.10

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ethics_raw"
    / "commonsense"
    / "train.csv"
)

TRAIN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ethics_dataset_train.csv"
)

VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ethics_dataset_val.csv"
)

REQUIRED_COLUMNS = {
    "label",
    "input",
}


# ===================== Data loading and validation =====================

def load_and_validate_dataset(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load and validate the raw ETHICS Commonsense dataset.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the raw training CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset containing the columns ``label`` and ``text``.

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.

    ValueError
        If required columns are missing or no valid samples remain.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {input_path}"
        )

    dataframe = pd.read_csv(
        input_path,
        encoding="utf-8",
    )

    print(
        "Available columns: "
        f"{list(dataframe.columns)}"
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The input dataset is missing the following "
            f"required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe[
        [
            "label",
            "input",
        ]
    ].copy()

    dataframe.rename(
        columns={
            "input": "text",
        },
        inplace=True,
    )

    dataframe.dropna(
        subset=[
            "label",
            "text",
        ],
        inplace=True,
    )

    dataframe["text"] = (
        dataframe["text"]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["text"] != ""
    ]

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    dataframe.dropna(
        subset=["label"],
        inplace=True,
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

    valid_labels = set(
        dataframe["label"].unique()
    )

    if not valid_labels.issubset({0, 1}):
        raise ValueError(
            "The label column must contain only binary "
            f"values 0 and 1. Found: {sorted(valid_labels)}"
        )

    dataframe.drop_duplicates(
        subset=[
            "text",
            "label",
        ],
        inplace=True,
    )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    if dataframe.empty:
        raise ValueError(
            "No valid samples remain after preprocessing."
        )

    label_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    if len(label_counts) < 2:
        raise ValueError(
            "At least two label classes are required "
            "for stratified splitting."
        )

    print(
        f"Valid samples after preprocessing: "
        f"{len(dataframe):,}"
    )

    print(
        "Class distribution:"
    )

    for label, count in label_counts.items():
        print(
            f"  Label {label}: {count:,}"
        )

    return dataframe


# ===================== Dataset splitting =====================

def split_dataset(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and validation sets.

    A stratified split is used to preserve the label distribution in
    both subsets.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Cleaned dataset containing ``label`` and ``text``.

    Returns
    -------
    tuple of pandas.DataFrame
        Training and validation datasets.
    """
    train_dataframe, validation_dataframe = (
        train_test_split(
            dataframe,
            test_size=VALIDATION_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=dataframe["label"],
        )
    )

    train_dataframe = (
        train_dataframe
        .sample(
            frac=1.0,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    validation_dataframe = (
        validation_dataframe
        .sample(
            frac=1.0,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    return (
        train_dataframe,
        validation_dataframe,
    )


# ===================== Output utilities =====================

def save_dataset(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a processed dataset as a UTF-8 encoded CSV file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to save.

    output_path : pathlib.Path
        Destination CSV path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Saved dataset: {output_path}"
    )


# ===================== Main workflow =====================

def main() -> None:
    """
    Execute the complete preprocessing workflow.
    """
    print(
        "Starting ETHICS dataset preprocessing."
    )

    dataframe = load_and_validate_dataset(
        INPUT_PATH
    )

    train_dataframe, validation_dataframe = (
        split_dataset(dataframe)
    )

    save_dataset(
        train_dataframe,
        TRAIN_OUTPUT_PATH,
    )

    save_dataset(
        validation_dataframe,
        VALIDATION_OUTPUT_PATH,
    )

    print(
        "\nDataset preprocessing completed successfully."
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Validation ratio: {VALIDATION_RATIO:.0%}"
    )

    print(
        f"Training samples: {len(train_dataframe):,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataframe):,}"
    )

    print(
        "\nTraining-set class distribution:"
    )

    for label, count in (
        train_dataframe["label"]
        .value_counts()
        .sort_index()
        .items()
    ):
        print(
            f"  Label {label}: {count:,}"
        )

    print(
        "\nValidation-set class distribution:"
    )

    for label, count in (
        validation_dataframe["label"]
        .value_counts()
        .sort_index()
        .items()
    ):
        print(
            f"  Label {label}: {count:,}"
        )


if __name__ == "__main__":
    main()
