"""
plot_figure5_clean.py

Purpose
-------
Generate publication-ready experimental figures for the multi-task BERT
framework.

All labels, annotations, legends, comments, and console messages are written
in English to support reproducibility and international peer review.

Generated files
---------------
Figure 2:
    Main F1-score heatmap.

Figure 3:
    Ablation study bar chart.

Figure 4:
    Training and validation loss curves.

Figure 5:
    Task-level F1-score improvement chart.

Figure 6:
    Token-level SHAP contribution chart.

Figure S1:
    Inter-task collaboration heatmap.

Output formats
--------------
PNG, PDF, and SVG.

Output directory
----------------
outputs/figs_plosone/
"""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ===================== Global configuration =====================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figs_plosone"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_FORMATS = ("png", "pdf", "svg")
PNG_DPI = 600

TASKS = [
    "Commonsense",
    "Deontology",
    "Justice",
    "Virtue",
    "Utilitarianism",
]

MODELS = [
    "BERT_MT",
    "BERT_ST",
    "DistilBERT_ST",
    "LogReg",
    "RoBERTa_ST",
    "SVM_TFIDF",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": PNG_DPI,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


# ===================== Validation utilities =====================

def validate_matrix(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    matrix_name: str,
) -> None:
    """
    Validate the dimensions and numeric values of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to validate.

    row_labels : sequence of str
        Labels corresponding to matrix rows.

    column_labels : sequence of str
        Labels corresponding to matrix columns.

    matrix_name : str
        Human-readable matrix name used in error messages.

    Raises
    ------
    ValueError
        If the matrix shape is inconsistent or contains non-finite values.
    """
    expected_shape = (
        len(row_labels),
        len(column_labels),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"{matrix_name} has shape {matrix.shape}, "
            f"but expected {expected_shape}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{matrix_name} contains non-finite numeric values."
        )


def validate_equal_lengths(
    labels: Sequence[str],
    values: Sequence[float],
    data_name: str,
) -> None:
    """
    Confirm that labels and numeric values contain the same number of items.

    Parameters
    ----------
    labels : sequence of str
        Category labels.

    values : sequence of float
        Numeric values.

    data_name : str
        Human-readable data name used in error messages.

    Raises
    ------
    ValueError
        If the sequence lengths differ.
    """
    if len(labels) != len(values):
        raise ValueError(
            f"{data_name} contains {len(labels)} labels "
            f"but {len(values)} values."
        )


# ===================== Figure saving =====================

def save_figure(
    figure: plt.Figure,
    file_stem: str,
) -> None:
    """
    Save a Matplotlib figure in PNG, PDF, and SVG formats.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.

    file_stem : str
        Output filename without extension.
    """
    for extension in FIGURE_FORMATS:
        output_path = OUTPUT_DIR / f"{file_stem}.{extension}"

        save_options = {
            "bbox_inches": "tight",
        }

        if extension == "png":
            save_options["dpi"] = PNG_DPI

        figure.savefig(
            output_path,
            **save_options,
        )

        print(f"[Figure] Saved: {output_path}")

    plt.close(figure)


# ===================== Figure 2 =====================

def plot_main_f1_heatmap() -> None:
    """
    Generate the main F1-score heatmap across models and ethical tasks.
    """
    main_f1 = np.array(
        [
            [0.80, 0.73, 0.70, 0.60, 0.75, 0.63],
            [0.52, 0.47, 0.45, 0.40, 0.48, 0.42],
            [0.79, 0.72, 0.68, 0.58, 0.74, 0.60],
            [0.64, 0.57, 0.55, 0.50, 0.59, 0.52],
            [0.68, 0.60, 0.58, 0.52, 0.62, 0.54],
        ],
        dtype=float,
    )

    validate_matrix(
        main_f1,
        TASKS,
        MODELS,
        "Main F1-score matrix",
    )

    figure, axis = plt.subplots(
        figsize=(9.5, 5.6)
    )

    image = axis.imshow(
        main_f1,
        cmap="YlGnBu",
        vmin=0.40,
        vmax=0.80,
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(len(MODELS))
    )
    axis.set_yticks(
        np.arange(len(TASKS))
    )

    axis.set_xticklabels(
        MODELS,
        rotation=0,
    )
    axis.set_yticklabels(
        TASKS
    )

    axis.set_xlabel("Model")
    axis.set_ylabel("Ethical task")

    for row_index in range(main_f1.shape[0]):
        for column_index in range(main_f1.shape[1]):
            value = main_f1[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value >= 0.62
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=11,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label("F1-score")

    figure.tight_layout()

    save_figure(
        figure,
        "Figure2_main_f1_heatmap_no_title",
    )


# ===================== Figure 3 =====================

def plot_ablation_results() -> None:
    """
    Generate the ablation study bar chart.
    """
    ablation_results = {
        "BERT_MT_full": [
            0.80,
            0.52,
            0.79,
            0.64,
            0.68,
        ],
        "BERT_MT_no_attention": [
            0.79,
            0.51,
            0.77,
            0.62,
            0.67,
        ],
        "BERT_MT_no_shap": [
            0.79,
            0.51,
            0.77,
            0.63,
            0.66,
        ],
        "BERT_ST_best": [
            0.73,
            0.47,
            0.72,
            0.57,
            0.60,
        ],
    }

    for model_name, values in ablation_results.items():
        validate_equal_lengths(
            TASKS,
            values,
            f"Ablation results for {model_name}",
        )

    x_positions = np.arange(
        len(TASKS)
    )
    bar_width = 0.18

    figure, axis = plt.subplots(
        figsize=(10.0, 5.4)
    )

    for model_index, (
        model_name,
        values,
    ) in enumerate(
        ablation_results.items()
    ):
        shifted_positions = (
            x_positions
            + (model_index - 1.5)
            * bar_width
        )

        axis.bar(
            shifted_positions,
            values,
            bar_width,
            label=model_name,
        )

    axis.set_ylabel("F1-score")
    axis.set_xlabel("Ethical task")

    axis.set_xticks(
        x_positions
    )
    axis.set_xticklabels(
        TASKS
    )

    axis.set_ylim(
        0.30,
        0.90,
    )

    axis.legend(
        title="Model variant",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "Figure3_ablation_f1_bar_no_title",
    )


# ===================== Figure 4 =====================

def plot_training_curves() -> None:
    """
    Generate training and validation loss curves.
    """
    epochs = np.array(
        [1, 2, 3, 4, 5],
        dtype=int,
    )

    train_mt = np.array(
        [0.90, 0.54, 0.47, 0.42, 0.36],
        dtype=float,
    )

    validation_mt = np.array(
        [1.04, 0.70, 0.49, 0.45, 0.42],
        dtype=float,
    )

    train_st = np.array(
        [0.90, 0.62, 0.48, 0.40, 0.35],
        dtype=float,
    )

    validation_st = np.array(
        [1.06, 0.70, 0.59, 0.48, 0.47],
        dtype=float,
    )

    for series_name, values in {
        "BERT_MT training loss": train_mt,
        "BERT_MT validation loss": validation_mt,
        "BERT_ST training loss": train_st,
        "BERT_ST validation loss": validation_st,
    }.items():
        validate_equal_lengths(
            epochs,
            values,
            series_name,
        )

    figure, axis = plt.subplots(
        figsize=(9.0, 5.4)
    )

    axis.plot(
        epochs,
        train_mt,
        marker="o",
        linewidth=2.2,
        label="Training (BERT_MT)",
    )

    axis.plot(
        epochs,
        validation_mt,
        marker="o",
        linestyle="--",
        linewidth=2.2,
        label="Validation (BERT_MT)",
    )

    axis.plot(
        epochs,
        train_st,
        marker="s",
        linewidth=2.2,
        label="Training (BERT_ST)",
    )

    axis.plot(
        epochs,
        validation_st,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label="Validation (BERT_ST)",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")

    axis.set_xticks(
        epochs
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    axis.legend(
        frameon=True
    )

    figure.tight_layout()

    save_figure(
        figure,
        "Figure4_training_validation_loss_no_title",
    )


# ===================== Figure 5 =====================

def plot_task_level_improvement() -> None:
    """
    Generate task-level relative F1-score improvement bars.
    """
    gains = np.array(
        [
            9.6,
            10.6,
            9.7,
            12.3,
            13.3,
        ],
        dtype=float,
    )

    validate_equal_lengths(
        TASKS,
        gains,
        "Task-level improvements",
    )

    y_positions = np.arange(
        len(TASKS)
    )

    figure, axis = plt.subplots(
        figsize=(8.8, 5.2)
    )

    bars = axis.barh(
        y_positions,
        gains,
    )

    axis.set_yticks(
        y_positions
    )
    axis.set_yticklabels(
        TASKS
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Relative F1-score improvement (%)"
    )
    axis.set_ylabel(
        "Ethical task"
    )

    axis.set_xlim(
        0,
        14.5,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.35,
    )

    for bar, value in zip(
        bars,
        gains,
    ):
        axis.text(
            value + 0.2,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1f}%",
            verticalalignment="center",
            fontsize=12,
        )

    figure.tight_layout()

    save_figure(
        figure,
        "Figure5_task_level_improvement_no_title",
    )


# ===================== Figure 6 =====================

def plot_shap_values() -> None:
    """
    Generate token-level SHAP contribution bars.
    """
    tokens = [
        "others",
        "help",
        "costs",
        "even",
        "you",
        "it",
        "if",
        "time",
    ]

    shap_values = np.array(
        [
            0.22,
            0.18,
            -0.10,
            0.05,
            -0.03,
            -0.02,
            0.01,
            0.00,
        ],
        dtype=float,
    )

    validate_equal_lengths(
        tokens,
        shap_values,
        "Token-level SHAP values",
    )

    y_positions = np.arange(
        len(tokens)
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 5.2)
    )

    axis.barh(
        y_positions,
        shap_values,
    )

    axis.axvline(
        0,
        color="black",
        linewidth=1.2,
    )

    axis.set_yticks(
        y_positions
    )
    axis.set_yticklabels(
        tokens
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "SHAP value"
    )
    axis.set_ylabel(
        "Token"
    )

    axis.set_xlim(
        -0.115,
        0.235,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.30,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "Figure6_token_level_shap_no_title",
    )


# ===================== Supplementary Figure S1 =====================

def plot_task_collaboration_heatmap() -> None:
    """
    Generate the supplementary inter-task collaboration heatmap.
    """
    collaboration_matrix = np.array(
        [
            [7.0, 6.0, 7.5, 6.7, 7.3],
            [5.8, 5.0, 5.3, 6.2, 6.6],
            [7.0, 5.9, 7.0, 6.3, 7.3],
            [6.8, 5.6, 6.9, 7.0, 7.7],
            [8.4, 6.6, 7.6, 7.5, 8.0],
        ],
        dtype=float,
    )

    validate_matrix(
        collaboration_matrix,
        TASKS,
        TASKS,
        "Inter-task collaboration matrix",
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 6.0)
    )

    image = axis.imshow(
        collaboration_matrix,
        cmap="RdYlGn",
        vmin=5.0,
        vmax=8.5,
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(len(TASKS))
    )
    axis.set_yticks(
        np.arange(len(TASKS))
    )

    axis.set_xticklabels(
        TASKS,
        rotation=45,
        horizontalalignment="right",
    )
    axis.set_yticklabels(
        TASKS
    )

    axis.set_xlabel(
        "Target task"
    )
    axis.set_ylabel(
        "Source task"
    )

    for row_index in range(
        collaboration_matrix.shape[0]
    ):
        for column_index in range(
            collaboration_matrix.shape[1]
        ):
            value = collaboration_matrix[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value < 5.7
                or value > 8.0
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=11,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Relative gain (%)"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "FigureS1_task_collaboration_heatmap_no_title",
    )


# ===================== Main workflow =====================

def main() -> None:
    """
    Generate all publication-ready experimental figures.
    """
    print(
        "Generating publication-ready figures."
    )

    plot_main_f1_heatmap()
    plot_ablation_results()
    plot_training_curves()
    plot_task_level_improvement()
    plot_shap_values()
    plot_task_collaboration_heatmap()

    print(
        "\nAll figures were generated successfully."
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
