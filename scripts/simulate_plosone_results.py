# -*- coding: utf-8 -*-
"""
simulate_plosone_results.py

用途：
- 为“多任务 BERT 伦理文本分类 + 可解释性”论文模拟一整套实验结果
- 自动生成图（PNG）和表（CSV + Markdown），占位用来写 PLOS ONE 论文
- 后续可用真实实验结果替换本脚本中的模拟数据

依赖：
    pip install numpy pandas matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 全局配置 =====================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR_FIGS = "outputs/figs_plosone"
OUTPUT_DIR_TABLES = "outputs/tables_plosone"

os.makedirs(OUTPUT_DIR_FIGS, exist_ok=True)
os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)

# 五个伦理任务（与你论文保持一致）
TASKS = ["Commonsense", "Deontology", "Justice", "Virtue", "Utilitarian"]

# 主要对比的模型
MODELS_MAIN = [
    "LogReg",
    "SVM_TFIDF",
    "DistilBERT_ST",
    "BERT_ST",
    "RoBERTa_ST",
    "BERT_MT"  # 你的多任务模型（ours）
]

# 消融实验版本（可在论文 Methods 里定义）
MODELS_ABLATION = [
    "BERT_MT_full",          # 完整模型（带 Attention + SHAP）
    "BERT_MT_no_attention",  # 去掉 Attention 解释模块
    "BERT_MT_no_shap",       # 去掉 SHAP 解释模块
    "BERT_ST_best"           # 性能最好单任务 BERT，用作对照
]

# ===================== 1. 构造模拟指标数据 =====================

def simulate_main_results():
    """
    模拟多任务 BERT 与各 baseline 在五个任务上的指标（Accuracy / F1 / AUC）。
    数值是“看起来合理”的占位数据，你可以按需要微调。
    返回：长表 DataFrame，列包含：
        model, task, acc, f1, auc
    """

    base_f1 = {
        "LogReg":        [0.60, 0.40, 0.58, 0.50, 0.52],
        "SVM_TFIDF":     [0.63, 0.42, 0.60, 0.52, 0.54],
        "DistilBERT_ST": [0.70, 0.45, 0.68, 0.55, 0.58],
        "BERT_ST":       [0.73, 0.47, 0.72, 0.57, 0.60],
        "RoBERTa_ST":    [0.75, 0.48, 0.74, 0.59, 0.62],
        "BERT_MT":       [0.80, 0.52, 0.79, 0.64, 0.68],  # ours，多任务略有优势
    }

    rows = []
    for model in MODELS_MAIN:
        for i, task in enumerate(TASKS):
            f1 = base_f1[model][i]
            # 用 F1 附近随机小振动生成 ACC / AUC
            acc = np.clip(f1 + np.random.normal(0.05, 0.02), 0.5, 0.95)
            auc = np.clip(f1 + np.random.normal(0.08, 0.02), 0.5, 0.99)

            rows.append({
                "model": model,
                "task": task,
                "acc": round(acc, 3),
                "f1": round(f1, 3),
                "auc": round(auc, 3),
            })

    df = pd.DataFrame(rows)
    return df


def simulate_ablation_results(main_df):
    """
    模拟消融实验结果：
    - BERT_MT_full 与去掉 Attention / 去掉 SHAP 对比
    - 再加一个 BERT_ST_best 作为参照
    """
    bert_mt = main_df[main_df["model"] == "BERT_MT"].copy()
    bert_mt.rename(columns={"model": "base_model"}, inplace=True)

    rows = []
    for _, row in bert_mt.iterrows():
        task = row["task"]
        f1_base = row["f1"]

        # 完整模型：等于 BERT_MT
        rows.append({
            "model": "BERT_MT_full",
            "task": task,
            "f1": f1_base,
        })

        # 去掉 Attention：略微变差
        rows.append({
            "model": "BERT_MT_no_attention",
            "task": task,
            "f1": round(f1_base - np.random.uniform(0.01, 0.02), 3),
        })

        # 去掉 SHAP：也略微变差
        rows.append({
            "model": "BERT_MT_no_shap",
            "task": task,
            "f1": round(f1_base - np.random.uniform(0.01, 0.02), 3),
        })

        # 单任务 BERT 作为对照
        bert_st_f1 = main_df[
            (main_df["model"] == "BERT_ST") & (main_df["task"] == task)
        ]["f1"].values[0]

        rows.append({
            "model": "BERT_ST_best",
            "task": task,
            "f1": bert_st_f1,
        })

    df_ablation = pd.DataFrame(rows)
    return df_ablation


def simulate_training_curves(num_epochs=5):
    """
    模拟训练/验证损失曲线：
    - 生成一个逐 epoch 下降的曲线，并带一点随机波动
    - 同时给单任务/多任务两条曲线，方便比较
    """
    epochs = np.arange(1, num_epochs + 1)
    train_loss_mt = []
    val_loss_mt = []
    train_loss_st = []
    val_loss_st = []

    base_train_start = 0.9
    base_val_start = 1.0

    for e in epochs:
        # 假装 Multi-task 收敛稍微更好一点
        tl_mt = base_train_start / (e ** 0.6) + np.random.normal(0, 0.02)
        vl_mt = base_val_start / (e ** 0.55) + np.random.normal(0, 0.03)

        tl_st = base_train_start / (e ** 0.55) + np.random.normal(0, 0.03)
        vl_st = base_val_start / (e ** 0.5) + np.random.normal(0, 0.04)

        train_loss_mt.append(max(tl_mt, 0.05))
        val_loss_mt.append(max(vl_mt, 0.05))
        train_loss_st.append(max(tl_st, 0.05))
        val_loss_st.append(max(vl_st, 0.05))

    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss_mt": np.round(train_loss_mt, 3),
        "val_loss_mt": np.round(val_loss_mt, 3),
        "train_loss_st": np.round(train_loss_st, 3),
        "val_loss_st": np.round(val_loss_st, 3),
    })
    return df


def simulate_task_correlation(main_df):
    """
    模拟任务间的“协同提升热力图”：
    - 以单任务与多任务 F1 差作为基础
    - 非对角线加一点随机扰动，用于展示“跨任务共享”的效果
    """
    improvements = {}

    for task in TASKS:
        f1_mt = main_df[(main_df["model"] == "BERT_MT") & (main_df["task"] == task)]["f1"].values[0]
        f1_st = main_df[(main_df["model"] == "BERT_ST") & (main_df["task"] == task)]["f1"].values[0]
        delta = (f1_mt - f1_st) * 100  # 百分比
        improvements[task] = delta

    mat = np.zeros((len(TASKS), len(TASKS)))
    for i, ti in enumerate(TASKS):
        for j, tj in enumerate(TASKS):
            if i == j:
                mat[i, j] = improvements[ti]
            else:
                mat[i, j] = (improvements[ti] + improvements[tj]) / 2 + np.random.normal(0, 0.5)

    df_corr = pd.DataFrame(mat, index=TASKS, columns=TASKS)
    return df_corr.round(2)


def simulate_shap_example():
    """
    模拟一个“SHAP token 贡献示意图”所需的数据：
    - 给出一句假想文本
    - 每个 token 一个 SHAP 值（正/负贡献）
    """
    tokens = ["help", "others", "even", "if", "it", "costs", "you", "time"]
    shap_values = [0.18, 0.22, 0.05, 0.01, -0.02, -0.10, -0.03, 0.00]
    df = pd.DataFrame({
        "token": tokens,
        "shap_value": shap_values
    })
    return df


# ===================== 2. 图表 & 表格 函数 =====================

def save_table_as_csv_and_md(df, name):
    """
    保存 CSV + Markdown（无需 tabulate）
    """
    csv_path = os.path.join(OUTPUT_DIR_TABLES, f"{name}.csv")
    md_path = os.path.join(OUTPUT_DIR_TABLES, f"{name}.md")

    # 保存 CSV
    df.to_csv(csv_path, index=False)

    # 手动生成 Markdown 表格
    with open(md_path, "w", encoding="utf-8") as f:
        headers = list(df.columns)
        # header
        f.write("| " + " | ".join(headers) + " |\n")
        # separator
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        # rows
        for _, row in df.iterrows():
            f.write("| " + " | ".join(map(str, row.values)) + " |\n")

    print(f"[table] Saved: {csv_path}, {md_path}")


def plot_main_results_heatmap(main_df):
    """
    画任务 × 模型 的 F1 heatmap：
        Figure: F1 scores of different models across five ethical tasks.
    """
    pivot = main_df.pivot(index="task", columns="model", values="f1")
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("F1 scores of models across tasks (simulated)")
    plt.ylabel("Task")
    plt.xlabel("Model")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR_FIGS, "fig_main_f1_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[fig] Saved: {fig_path}")


def plot_ablation_bar(df_ablation):
    """
    画消融实验柱状图
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df_ablation,
        x="task",
        y="f1",
        hue="model"
    )
    plt.ylim(0.3, 0.9)
    plt.ylabel("F1 score")
    plt.title("Ablation study on multi-task model (simulated)")
    plt.legend(title="Model variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR_FIGS, "fig_ablation_f1_bar.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[fig] Saved: {fig_path}")


def plot_training_curves(df_train):
    """
    画训练/验证损失曲线图
    """
    plt.figure(figsize=(6, 4))
    plt.plot(df_train["epoch"], df_train["train_loss_mt"], marker="o", label="Train (BERT_MT)")
    plt.plot(df_train["epoch"], df_train["val_loss_mt"], marker="o", linestyle="--", label="Val (BERT_MT)")
    plt.plot(df_train["epoch"], df_train["train_loss_st"], marker="s", label="Train (BERT_ST)")
    plt.plot(df_train["epoch"], df_train["val_loss_st"], marker="s", linestyle="--", label="Val (BERT_ST)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss (simulated)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR_FIGS, "fig_training_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[fig] Saved: {fig_path}")


def plot_task_correlation_heatmap(df_corr):
    """
    画“任务间协同提升热力图”
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(df_corr, annot=True, fmt=".1f", cmap="RdYlGn")
    plt.title("Task-level relative gains of multi-task learning (%) (simulated)")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR_FIGS, "fig_task_collaboration_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[fig] Saved: {fig_path}")


def plot_shap_example(df_shap):
    """
    画一个简单的 SHAP token-贡献条形图
    """
    plt.figure(figsize=(6, 3))
    df_plot = df_shap.sort_values(by="shap_value", key=lambda x: np.abs(x), ascending=True)

    plt.barh(df_plot["token"], df_plot["shap_value"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("SHAP value")
    plt.title("Example of token-level SHAP values (simulated)")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR_FIGS, "fig_shap_token_example.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[fig] Saved: {fig_path}")


# ===================== 3. 主流程 =====================

def main():
    # 1) 主结果：不同模型在各任务上的指标
    df_main = simulate_main_results()
    save_table_as_csv_and_md(df_main, "table_main_results")

    # 平均 F1
    df_avg = df_main.groupby("model")["f1"].mean().reset_index().rename(columns={"f1": "mean_f1"})
    save_table_as_csv_and_md(df_avg, "table_main_results_avg")

    # 2) 消融实验
    df_ablation = simulate_ablation_results(df_main)
    save_table_as_csv_and_md(df_ablation, "table_ablation")

    # 3) 训练曲线
    df_train = simulate_training_curves(num_epochs=5)
    save_table_as_csv_and_md(df_train, "table_training_loss")

    # 4) 任务协同热力图数据
    df_corr = simulate_task_correlation(df_main)
    df_corr_out = df_corr.copy()
    df_corr_out.insert(0, "task", df_corr_out.index)
    save_table_as_csv_and_md(df_corr_out, "table_task_collaboration")

    # 5) SHAP 示例
    df_shap = simulate_shap_example()
    save_table_as_csv_and_md(df_shap, "table_shap_example")

    # —— 绘图 ——
    plot_main_results_heatmap(df_main)
    plot_ablation_bar(df_ablation)
    plot_training_curves(df_train)
    plot_task_correlation_heatmap(df_corr)
    plot_shap_example(df_shap)

    print("\nAll simulated figures and tables have been generated.")


if __name__ == "__main__":
    main()