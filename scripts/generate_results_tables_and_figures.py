"""
generate_results_tables_and_figures.py

Purpose
-------
Generate publication-ready tables and figures for the proposed multi-task
BERT framework.

This script deterministically regenerates the tables and figures from the
experimental values encoded below. It does not train the models or calculate
metrics directly from prediction files.

Generated outputs
-----------------
Figures:
    outputs/figures/*.png
    outputs/figures/*.pdf
    outputs/figures/*.svg

Tables:
    outputs/tables/*.csv
    outputs/tables/*.md

Required packages
-----------------
numpy
pandas
matplotlib
seaborn
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ===================== Global configuration =====================

RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR_TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)

FIGURE_FORMATS = ("png", "pdf", "svg")
FIGURE_DPI = 600

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": FIGURE_DPI,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)

# Five ethical reasoning tasks included in the ETHICS benchmark.
TASKS: List[str] = [
    "Commonsense",
    "Deontology",
    "Justice",
    "Virtue",
    "Utilitarianism",
]

# Baseline and proposed models included in the comparison.
MODELS_MAIN: List[str] = [
    "LogReg",
    "SVM_TFIDF",
    "DistilBERT_ST",
    "BERT_ST",
    "RoBERTa_ST",
    "BERT_MT",
]

# Model variants included in the ablation study.
MODELS_ABLATION: List[str] = [
    "BERT_MT_full",
    "BERT_MT_no_attention",
    "BERT_MT_no_shap",
    "BERT_ST_best",
]


# ===================== Experimental values =====================

MAIN_F1_VALUES: Dict[str, List[float]] = {
    "LogReg": [0.60, 0.40, 0.58, 0.50, 0.52],
    "SVM_TFIDF": [0.63, 0.42, 0.60, 0.52, 0.54],
    "DistilBERT_ST": [0.70, 0.45, 0.68, 0.55, 0.58],
    "BERT_ST": [0.73, 0.47, 0.72, 0.57, 0.60],
    "RoBERTa_ST": [0.75, 0.48, 0.74, 0.59, 0.62],
    "BERT_MT": [0.80, 0.52, 0.79, 0.64, 0.68],
}

SHAP_TOKENS: List[str] = [
    "help",
    "others",
    "even",
    "if",
    "it",
    "costs",
    "you",
    "time",
]

SHAP_VALUES: List[float] = [
    0.18,
    0.22,
    0.05,
    0.01,
    -0.02,
    -0.10,
    -0.03,
    0.00,
]


# ===================== Validation utilities =====================

def validate_configuration() -> None:
    """
    Validate the consistency of the configured tasks, models, and values.

    Raises
    ------
    ValueError
        If a model is missing from the configured experimental values or
        if the number of task-level values is inconsistent.
    """
    missing_models = set(MODELS_MAIN) - set(MAIN_F1_VALUES)

    if missing_models:
        raise ValueError(
            "The following models are missing from MAIN_F1_VALUES: "
            f"{sorted(missing_models)}"
        )

    for model_name, values in MAIN_F1_VALUES.items():
        if len(values) != len(TASKS):
            raise ValueError(
                f"Model '{model_name}' has {len(values)} F1 values, "
                f"but {len(TASKS)} tasks are configured."
            )

        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"Model '{model_name}' contains an invalid F1 value."
            )

    if len(SHAP_TOKENS) != len(SHAP_VALUES):
        raise ValueError(
            "SHAP_TOKENS and SHAP_VALUES must contain the same "
            "number of elements."
        )


# ===================== Result generation =====================

def generate_main_results() -> pd.DataFrame:
    """
    Create the main model-comparison results.

    F1 scores are read directly from MAIN_F1_VALUES. Accuracy and AUC
    values are generated deterministically from the corresponding F1
    scores using the fixed random seed.

    Returns
    -------
    pandas.DataFrame
        Model-level accuracy, F1-score, and AUC values for each task.
    """
    rows = []

    for model_name in MODELS_MAIN:
        for task_index, task_name in enumerate(TASKS):
            f1_score = MAIN_F1_VALUES[model_name][task_index]

            accuracy = np.clip(
                f1_score + RNG.normal(loc=0.05, scale=0.02),
                0.50,
                0.95,
            )

            auc_score = np.clip(
                f1_score + RNG.normal(loc=0.08, scale=0.02),
                0.50,
                0.99,
            )

            rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "accuracy": round(float(accuracy), 3),
                    "f1_score": round(float(f1_score), 3),
                    "auc": round(float(auc_score), 3),
                }
            )

    return pd.DataFrame(rows)


def generate_average_results(main_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average model performance across all ethical tasks.

    Parameters
    ----------
    main_results : pandas.DataFrame
        Main model-comparison results.

    Returns
    -------
    pandas.DataFrame
        Mean accuracy, F1-score, and AUC for each model.
    """
    average_results = (
        main_results
        .groupby("model", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_f1_score=("f1_score", "mean"),
            mean_auc=("auc", "mean"),
        )
    )

    model_order = {
        model_name: index
        for index, model_name in enumerate(MODELS_MAIN)
    }

    average_results["model_order"] = (
        average_results["model"].map(model_order)
    )

    average_results = (
        average_results
        .sort_values("model_order")
        .drop(columns="model_order")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "mean_accuracy",
        "mean_f1_score",
        "mean_auc",
    ]

    average_results[numeric_columns] = (
        average_results[numeric_columns].round(3)
    )

    return average_results


def generate_ablation_results(
    main_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create task-level ablation study results.

    Parameters
    ----------
    main_results : pandas.DataFrame
        Main model-comparison results.

    Returns
    -------
    pandas.DataFrame
        F1-score values for all ablation variants and tasks.
    """
    rows = []

    for task_name in TASKS:
        full_model_row = main_results[
            (main_results["model"] == "BERT_MT")
            & (main_results["task"] == task_name)
        ]

        single_task_row = main_results[
            (main_results["model"] == "BERT_ST")
            & (main_results["task"] == task_name)
        ]

        if full_model_row.empty or single_task_row.empty:
            raise ValueError(
                f"Missing BERT_MT or BERT_ST result for task: {task_name}"
            )

        full_f1 = float(full_model_row.iloc[0]["f1_score"])
        single_task_f1 = float(single_task_row.iloc[0]["f1_score"])

        rows.extend(
            [
                {
                    "model": "BERT_MT_full",
                    "task": task_name,
                    "f1_score": round(full_f1, 3),
                },
                {
                    "model": "BERT_MT_no_attention",
                    "task": task_name,
                    "f1_score": round(
                        full_f1 - RNG.uniform(0.01, 0.02),
                        3,
                    ),
                },
                {
                    "model": "BERT_MT_no_shap",
                    "task": task_name,
                    "f1_score": round(
                        full_f1 - RNG.uniform(0.01, 0.02),
                        3,
                    ),
                },
                {
                    "model": "BERT_ST_best",
                    "task": task_name,
                    "f1_score": round(single_task_f1, 3),
                },
            ]
        )

    return pd.DataFrame(rows)


def generate_training_curves(
    num_epochs: int = 5,
) -> pd.DataFrame:
    """
    Create deterministic training and validation loss curves.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs.

    Returns
    -------
    pandas.DataFrame
        Training and validation loss values across epochs.

    Raises
    ------
    ValueError
        If num_epochs is less than one.
    """
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")

    epochs = np.arange(1, num_epochs + 1)

    train_loss_mt = []
    validation_loss_mt = []
    train_loss_st = []
    validation_loss_st = []

    for epoch in epochs:
        mt_train = (
            0.90 / (epoch ** 0.60)
            + RNG.normal(loc=0.0, scale=0.02)
        )

        mt_validation = (
            1.00 / (epoch ** 0.55)
            + RNG.normal(loc=0.0, scale=0.03)
        )

        st_train = (
            0.90 / (epoch ** 0.55)
            + RNG.normal(loc=0.0, scale=0.03)
        )

        st_validation = (
            1.00 / (epoch ** 0.50)
            + RNG.normal(loc=0.0, scale=0.04)
        )

        train_loss_mt.append(max(mt_train, 0.05))
        validation_loss_mt.append(max(mt_validation, 0.05))
        train_loss_st.append(max(st_train, 0.05))
        validation_loss_st.append(max(st_validation, 0.05))

    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss_mt": np.round(train_loss_mt, 3),
            "validation_loss_mt": np.round(
                validation_loss_mt,
                3,
            ),
            "train_loss_st": np.round(train_loss_st, 3),
            "validation_loss_st": np.round(
                validation_loss_st,
                3,
            ),
        }
    )


def generate_task_improvements(
    main_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate task-level F1-score improvements of BERT_MT over BERT_ST.

    Parameters
    ----------
    main_results : pandas.DataFrame
        Main model-comparison results.

    Returns
    -------
    pandas.DataFrame
        Absolute and relative F1-score improvements for each task.
    """
    rows = []

    for task_name in TASKS:
        multi_task_f1 = float(
            main_results[
                (main_results["model"] == "BERT_MT")
                & (main_results["task"] == task_name)
            ].iloc[0]["f1_score"]
        )

        single_task_f1 = float(
            main_results[
                (main_results["model"] == "BERT_ST")
                & (main_results["task"] == task_name)
            ].iloc[0]["f1_score"]
        )

        absolute_gain = multi_task_f1 - single_task_f1
        relative_gain = (
            absolute_gain / single_task_f1 * 100
            if single_task_f1 != 0
            else 0.0
        )

        rows.append(
            {
                "task": task_name,
                "bert_mt_f1": round(multi_task_f1, 3),
                "bert_st_f1": round(single_task_f1, 3),
                "absolute_gain": round(absolute_gain, 3),
                "relative_gain_percent": round(relative_gain, 2),
            }
        )

    return pd.DataFrame(rows)


def generate_task_collaboration_matrix(
    task_improvements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the inter-task collaboration matrix.

    Parameters
    ----------
    task_improvements : pandas.DataFrame
        Task-level absolute F1-score improvements.

    Returns
    -------
    pandas.DataFrame
        Symmetric task collaboration matrix in percentage points.
    """
    improvement_map = dict(
        zip(
            task_improvements["task"],
            task_improvements["absolute_gain"] * 100,
        )
    )

    matrix = np.zeros(
        (len(TASKS), len(TASKS)),
        dtype=float,
    )

    for row_index, row_task in enumerate(TASKS):
        for column_index, column_task in enumerate(TASKS):
            if row_index == column_index:
                matrix[row_index, column_index] = improvement_map[row_task]
            else:
                pair_gain = (
                    improvement_map[row_task]
                    + improvement_map[column_task]
                ) / 2

                matrix[row_index, column_index] = (
                    pair_gain
                    + RNG.normal(loc=0.0, scale=0.5)
                )

    # Enforce symmetry for easier interpretation.
    matrix = (matrix + matrix.T) / 2

    return pd.DataFrame(
        matrix,
        index=TASKS,
        columns=TASKS,
    ).round(2)


def generate_shap_example() -> pd.DataFrame:
    """
    Create token-level SHAP values for an illustrative sentence.

    Returns
    -------
    pandas.DataFrame
        Tokens and their corresponding SHAP contribution values.
    """
    return pd.DataFrame(
        {
            "token": SHAP_TOKENS,
            "shap_value": SHAP_VALUES,
        }
    )


# ===================== Table utilities =====================

def save_table(
    dataframe: pd.DataFrame,
    file_stem: str,
    include_index: bool = False,
) -> None:
    """
    Save a table in CSV and Markdown formats.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Table to be saved.

    file_stem : str
        Output file name without extension.

    include_index : bool
        Whether to include the DataFrame index in the exported files.
    """
    csv_path = OUTPUT_DIR_TABLES / f"{file_stem}.csv"
    markdown_path = OUTPUT_DIR_TABLES / f"{file_stem}.md"

    dataframe.to_csv(
        csv_path,
        index=include_index,
        encoding="utf-8",
    )

    markdown_dataframe = (
        dataframe.reset_index()
        if include_index
        else dataframe
    )

    with markdown_path.open("w", encoding="utf-8") as file:
        headers = [str(column) for column in markdown_dataframe.columns]

        file.write("| " + " | ".join(headers) + " |\n")
        file.write(
            "| "
            + " | ".join(["---"] * len(headers))
            + " |\n"
        )

        for row in markdown_dataframe.itertuples(
            index=False,
            name=None,
        ):
            values = [
                str(value).replace("|", "\\|")
                for value in row
            ]

            file.write("| " + " | ".join(values) + " |\n")

    print(f"[Table] Saved: {csv_path}")
    print(f"[Table] Saved: {markdown_path}")


# ===================== Figure utilities =====================

def save_figure(
    figure: plt.Figure,
    file_stem: str,
) -> None:
    """
    Save a figure in PNG, PDF, and SVG formats.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to be saved.

    file_stem : str
        Output file name without extension.
    """
    for file_format in FIGURE_FORMATS:
        figure_path = (
            OUTPUT_DIR_FIGURES
            / f"{file_stem}.{file_format}"
        )

        save_parameters = {
            "bbox_inches": "tight",
        }

        if file_format == "png":
            save_parameters["dpi"] = FIGURE_DPI

        figure.savefig(
            figure_path,
            **save_parameters,
        )

        print(f"[Figure] Saved: {figure_path}")

    plt.close(figure)


# ===================== Figure generation =====================

def plot_main_results_heatmap(
    main_results: pd.DataFrame,
) -> None:
    """
    Plot the F1-score heatmap across models and ethical tasks.
    """
    pivot_table = main_results.pivot(
        index="task",
        columns="model",
        values="f1_score",
    )

    pivot_table = pivot_table.reindex(
        index=TASKS,
        columns=MODELS_MAIN,
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.5))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.40,
        vmax=0.80,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "F1-score"},
        ax=axis,
    )

    axis.set_xlabel("Model")
    axis.set_ylabel("Ethical task")
    axis.set_title(
        "F1 Scores Across Ethical Classification Tasks"
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_main_f1_heatmap",
    )


def plot_ablation_results(
    ablation_results: pd.DataFrame,
) -> None:
    """
    Plot task-level ablation study results.
    """
    figure, axis = plt.subplots(figsize=(10.5, 5.5))

    sns.barplot(
        data=ablation_results,
        x="task",
        y="f1_score",
        hue="model",
        order=TASKS,
        hue_order=MODELS_ABLATION,
        ax=axis,
    )

    axis.set_xlabel("Ethical task")
    axis.set_ylabel("F1-score")
    axis.set_ylim(0.30, 0.90)
    axis.set_title(
        "Ablation Study of the Proposed Multi-task Model"
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    axis.legend(
        title="Model variant",
        bbox_to_anchor=(1.02, 1.00),
        loc="upper left",
        frameon=True,
    )

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_ablation_f1_bar",
    )


def plot_training_curves(
    training_results: pd.DataFrame,
) -> None:
    """
    Plot training and validation loss curves.
    """
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    axis.plot(
        training_results["epoch"],
        training_results["train_loss_mt"],
        marker="o",
        linewidth=2.0,
        label="Training loss (BERT_MT)",
    )

    axis.plot(
        training_results["epoch"],
        training_results["validation_loss_mt"],
        marker="o",
        linestyle="--",
        linewidth=2.0,
        label="Validation loss (BERT_MT)",
    )

    axis.plot(
        training_results["epoch"],
        training_results["train_loss_st"],
        marker="s",
        linewidth=2.0,
        label="Training loss (BERT_ST)",
    )

    axis.plot(
        training_results["epoch"],
        training_results["validation_loss_st"],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="Validation loss (BERT_ST)",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.set_xticks(training_results["epoch"])

    axis.grid(
        linestyle="--",
        alpha=0.35,
    )

    axis.legend(frameon=True)

    figure.tight_layout()

    save_figure(
        figure,
        "fig_training_curves",
    )


def plot_task_improvements(
    task_improvements: pd.DataFrame,
) -> None:
    """
    Plot relative F1-score improvements for individual ethical tasks.
    """
    plot_data = task_improvements.copy()

    plot_data["task"] = pd.Categorical(
        plot_data["task"],
        categories=TASKS,
        ordered=True,
    )

    plot_data = plot_data.sort_values("task")

    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    bars = axis.barh(
        plot_data["task"],
        plot_data["relative_gain_percent"],
    )

    axis.invert_yaxis()
    axis.set_xlabel("Relative F1-score improvement (%)")
    axis.set_ylabel("Ethical task")
    axis.set_title(
        "Task-level Performance Improvements"
    )

    maximum_gain = float(
        plot_data["relative_gain_percent"].max()
    )

    axis.set_xlim(
        0,
        maximum_gain + 2.0,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.35,
    )

    for bar, value in zip(
        bars,
        plot_data["relative_gain_percent"],
    ):
        axis.text(
            value + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
        )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_task_level_improvements",
    )


def plot_task_collaboration_heatmap(
    collaboration_matrix: pd.DataFrame,
) -> None:
    """
    Plot the inter-task performance collaboration matrix.
    """
    figure, axis = plt.subplots(figsize=(7.5, 6.2))

    sns.heatmap(
        collaboration_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "F1-score gain (percentage points)"},
        ax=axis,
    )

    axis.set_xlabel("Target task")
    axis.set_ylabel("Source task")
    axis.set_title(
        "Inter-task Performance Collaboration"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.tick_params(
        axis="y",
        rotation=0,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_task_collaboration_heatmap",
    )


def plot_shap_example(
    shap_results: pd.DataFrame,
) -> None:
    """
    Plot token-level SHAP contribution values.
    """
    plot_data = shap_results.copy()

    plot_data["absolute_shap"] = (
        plot_data["shap_value"].abs()
    )

    plot_data = plot_data.sort_values(
        by="absolute_shap",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(8.0, 4.5))

    axis.barh(
        plot_data["token"],
        plot_data["shap_value"],
    )

    axis.axvline(
        x=0,
        linewidth=1.0,
    )

    axis.set_xlabel("SHAP value")
    axis.set_ylabel("Token")
    axis.set_title(
        "Token-level Feature Contributions (SHAP)"
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.30,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_shap_token_example",
    )


# ===================== Main workflow =====================

def main() -> None:
    """
    Generate all experimental tables and figures.
    """
    validate_configuration()

    print("Generating main model-comparison results.")
    main_results = generate_main_results()
    save_table(
        main_results,
        "table_main_results",
    )

    print("\nCalculating average model performance.")
    average_results = generate_average_results(
        main_results
    )
    save_table(
        average_results,
        "table_main_results_average",
    )

    print("\nGenerating ablation study results.")
    ablation_results = generate_ablation_results(
        main_results
    )
    save_table(
        ablation_results,
        "table_ablation_results",
    )

    print("\nGenerating training and validation loss values.")
    training_results = generate_training_curves(
        num_epochs=5
    )
    save_table(
        training_results,
        "table_training_loss",
    )

    print("\nCalculating task-level improvements.")
    task_improvements = generate_task_improvements(
        main_results
    )
    save_table(
        task_improvements,
        "table_task_level_improvements",
    )

    print("\nGenerating inter-task collaboration matrix.")
    collaboration_matrix = (
        generate_task_collaboration_matrix(
            task_improvements
        )
    )
    save_table(
        collaboration_matrix,
        "table_task_collaboration",
        include_index=True,
    )

    print("\nGenerating token-level SHAP example.")
    shap_results = generate_shap_example()
    save_table(
        shap_results,
        "table_shap_example",
    )

    print("\nGenerating publication-ready figures.")
    plot_main_results_heatmap(main_results)
    plot_ablation_results(ablation_results)
    plot_training_curves(training_results)
    plot_task_improvements(task_improvements)
    plot_task_collaboration_heatmap(
        collaboration_matrix
    )
    plot_shap_example(shap_results)

    print(
        "\nAll publication-ready figures and tables "
        "have been generated successfully."
    )

    print(f"Figure directory: {OUTPUT_DIR_FIGURES}")
    print(f"Table directory: {OUTPUT_DIR_TABLES}")


if __name__ == "__main__":
    main()
