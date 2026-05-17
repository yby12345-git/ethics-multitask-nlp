import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "outputs/figs_plosone"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": True,
    "axes.spines.right": True
})

def save_all(name):
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()

# =========================
# Figure 2: Main F1 heatmap
# =========================
tasks = ["Commonsense", "Deontology", "Justice", "Utilitarian", "Virtue"]
models = ["BERT_MT", "BERT_ST", "DistilBERT_ST", "LogReg", "RoBERTa_ST", "SVM_TFIDF"]
main_f1 = np.array([
    [0.80, 0.73, 0.70, 0.60, 0.75, 0.63],
    [0.52, 0.47, 0.45, 0.40, 0.48, 0.42],
    [0.79, 0.72, 0.68, 0.58, 0.74, 0.60],
    [0.68, 0.60, 0.58, 0.52, 0.62, 0.54],
    [0.64, 0.57, 0.55, 0.50, 0.59, 0.52],
])

fig, ax = plt.subplots(figsize=(9.5, 5.6))
im = ax.imshow(main_f1, cmap="YlGnBu", vmin=0.40, vmax=0.80)
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(tasks)))
ax.set_xticklabels(models, rotation=0)
ax.set_yticklabels(tasks)
ax.set_xlabel("Model")
ax.set_ylabel("Task")
for i in range(main_f1.shape[0]):
    for j in range(main_f1.shape[1]):
        color = "white" if main_f1[i, j] >= 0.62 else "black"
        ax.text(j, i, f"{main_f1[i, j]:.3f}", ha="center", va="center", color=color, fontsize=11)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("F1-score")
save_all("Figure2_main_f1_heatmap_no_title")

# =========================
# Figure 3: Ablation bar
# =========================
tasks_ab = ["Commonsense", "Deontology", "Justice", "Virtue", "Utilitarian"]
ablation = {
    "BERT_MT_full": [0.80, 0.52, 0.79, 0.64, 0.68],
    "BERT_MT_no_attention": [0.79, 0.51, 0.77, 0.62, 0.67],
    "BERT_MT_no_shap": [0.79, 0.51, 0.77, 0.63, 0.66],
    "BERT_ST_best": [0.73, 0.47, 0.72, 0.57, 0.60],
}
x = np.arange(len(tasks_ab))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5.4))
for idx, (label, values) in enumerate(ablation.items()):
    ax.bar(x + (idx - 1.5) * width, values, width, label=label)
ax.set_ylabel("F1-score")
ax.set_xlabel("Task")
ax.set_xticks(x)
ax.set_xticklabels(tasks_ab)
ax.set_ylim(0.30, 0.90)
ax.legend(title="Model variant", frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.grid(axis="y", linestyle="--", alpha=0.35)
save_all("Figure3_ablation_f1_bar_no_title")

# =========================
# Figure 4: Training curves
# =========================
epochs = np.array([1, 2, 3, 4, 5])
train_mt = np.array([0.90, 0.54, 0.47, 0.42, 0.36])
val_mt = np.array([1.04, 0.70, 0.49, 0.45, 0.42])
train_st = np.array([0.90, 0.62, 0.48, 0.40, 0.35])
val_st = np.array([1.06, 0.70, 0.59, 0.48, 0.47])

fig, ax = plt.subplots(figsize=(9, 5.4))
ax.plot(epochs, train_mt, marker="o", linewidth=2.2, label="Train (BERT_MT)")
ax.plot(epochs, val_mt, marker="o", linestyle="--", linewidth=2.2, label="Val (BERT_MT)")
ax.plot(epochs, train_st, marker="s", linewidth=2.2, label="Train (BERT_ST)")
ax.plot(epochs, val_st, marker="s", linestyle="--", linewidth=2.2, label="Val (BERT_ST)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(epochs)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(frameon=True)
save_all("Figure4_training_validation_loss_no_title")

# =========================
# Figure 5: Task-level improvement
# =========================
tasks_gain = ["Commonsense", "Deontology", "Justice", "Virtue", "Utilitarian"]
gains = np.array([9.6, 10.6, 9.7, 12.3, 13.3])
y = np.arange(len(tasks_gain))

fig, ax = plt.subplots(figsize=(8.8, 5.2))
bars = ax.barh(y, gains)
ax.set_yticks(y)
ax.set_yticklabels(tasks_gain)
ax.invert_yaxis()
ax.set_xlabel("F1-score improvement (%)")
ax.set_ylabel("Ethical task")
ax.set_xlim(0, 14)
ax.grid(axis="x", linestyle="--", alpha=0.35)
for bar, value in zip(bars, gains):
    ax.text(value + 0.2, bar.get_y() + bar.get_height()/2, f"{value:.1f}%", va="center", fontsize=12)
save_all("Figure5_task_level_improvement_no_title")

# =========================
# Figure 6: SHAP token values
# =========================
tokens = ["others", "help", "costs", "even", "you", "it", "if", "time"]
shap_values = np.array([0.22, 0.18, -0.10, 0.05, -0.03, -0.02, 0.01, 0.00])
y = np.arange(len(tokens))

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(y, shap_values)
ax.axvline(0, color="black", linewidth=1.2)
ax.set_yticks(y)
ax.set_yticklabels(tokens)
ax.invert_yaxis()
ax.set_xlabel("SHAP value")
ax.set_xlim(-0.115, 0.235)
ax.grid(axis="x", linestyle="--", alpha=0.30)
save_all("Figure6_token_level_shap_no_title")

# =========================
# Supplementary/inter-task heatmap
# =========================
collab_tasks = ["Commonsense", "Deontology", "Justice", "Virtue", "Utilitarian"]
collab = np.array([
    [7.0, 6.0, 7.5, 6.7, 7.3],
    [5.8, 5.0, 5.3, 6.2, 6.6],
    [7.0, 5.9, 7.0, 6.3, 7.3],
    [6.8, 5.6, 6.9, 7.0, 7.7],
    [8.4, 6.6, 7.6, 7.5, 8.0],
])

fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(collab, cmap="RdYlGn", vmin=5.0, vmax=8.5)
ax.set_xticks(np.arange(len(collab_tasks)))
ax.set_yticks(np.arange(len(collab_tasks)))
ax.set_xticklabels(collab_tasks, rotation=90)
ax.set_yticklabels(collab_tasks)
for i in range(collab.shape[0]):
    for j in range(collab.shape[1]):
        color = "white" if collab[i, j] < 5.7 or collab[i, j] > 8.0 else "black"
        ax.text(j, i, f"{collab[i, j]:.1f}", ha="center", va="center", color=color, fontsize=11)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Relative gain (%)")
save_all("FigureS1_task_collaboration_heatmap_no_title")

print(f"Done. Figures saved to: {OUT_DIR}")
