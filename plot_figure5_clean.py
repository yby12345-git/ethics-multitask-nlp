import matplotlib.pyplot as plt
import numpy as np

# 数据
tasks = ["Commonsense", "Deontology", "Justice", "Virtue", "Utilitarian"]
improvement = [9.6, 10.6, 9.7, 12.3, 13.3]

# 反转顺序（让最高在上）
tasks = tasks[::-1]
improvement = improvement[::-1]

# 颜色（渐变蓝）
colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(tasks)))

# 创建图
fig, ax = plt.subplots(figsize=(7, 4.5))

# 横向条形图
bars = ax.barh(tasks, improvement, color=colors, edgecolor='black', linewidth=0.6)

# 数值标注
for i, v in enumerate(improvement):
    ax.text(v + 0.2, i, f"{v:.1f}%", va='center', fontsize=10)

# 坐标轴
ax.set_xlabel("F1-score Improvement (%)", fontsize=11)
ax.set_ylabel("Ethical Task", fontsize=11)

# 去掉上边框和右边框（关键）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保留左和下轴线
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# 网格（淡一点）
ax.grid(axis='x', linestyle='--', alpha=0.4)

# 去标题（不写）
plt.tight_layout()

# 保存
plt.savefig("Figure5_clean.png", dpi=300, bbox_inches='tight')
plt.savefig("Figure5_clean.pdf", bbox_inches='tight')

print("Saved: Figure5_clean.png / .pdf")
