import matplotlib.pyplot as plt
import numpy as np

# 设置字体大小
# plt.rcParams.update({'font.size': 11})

methods = [ r'$\mathrm{CoOp}_{\mathrm{MCM}}$',  r'$\mathrm{CoOp}_{\mathrm{GL}}$',  r'$\mathrm{LoCoOp}_{\mathrm{MCM}}$', r'$\mathrm{LoCoOp}_{\mathrm{GL}}$','IDLike', 'SCT', r'$\mathrm{FA}_{\mathrm{MCM}}(Ours)$', r'$\mathrm{FA}_{\mathrm{GL}}(Ours)$']
markers = ['o', 's', '^', 'v','p', 'h', 'D', '*']  # Different markers for each method
colors = ['#2ca02c', '#8c564b', '#9467bd', '#bcbd22', '#e377c2', '#7f7f7f', '#d62728', '#17becf']  # Different colors for each method
sizes = [70, 70, 70, 70, 70, 70, 50, 95]  # Different sizes for each method



# Data for Subplot (a)
fpr95_a = [44.24, 37.59, 38.85, 32.53, 36.36, 31.63, 31.63, 27.12]
auroc_a = [90.03, 89.91, 91.59, 92.17, 91.93, 92.01, 93.36, 93.49]

# Data for Subplot (b)
fpr95_b = [37.89, 31.81, 34.47, 28.94, 41.18, 27.27, 29.60, 25.69]
auroc_b = [91.52, 91.51, 92.49, 93.13, 90.53, 93.31, 93.72, 93.76]


fig, axs = plt.subplots(2, 1, figsize=(8, 8))
# 先绘制网格线
axs[0].grid(True)
axs[0].set_axisbelow(True)  # 确保网格线在图形和刻度线下方
# 获取当前的网格线并设置透明度
[grid.set_alpha(0.3) for grid in axs[0].get_xgridlines() + axs[0].get_ygridlines()]

axs[1].grid(True)
axs[1].set_axisbelow(True)
[grid.set_alpha(0.3) for grid in axs[1].get_xgridlines() + axs[1].get_ygridlines()]

# Subplot (a)
for i, (x, y) in enumerate(zip(fpr95_a, auroc_a)):
    axs[0].scatter(x, y, marker=markers[i], label=methods[i],zorder=2, color=colors[i], s=sizes[i])
axs[0].set_xlabel('FPR95(%) ↓', fontsize=12)
axs[0].set_ylabel('AUROC(%) ↑', fontsize=13)

# Add legend and title below FPR95 for subplot (a)
axs[0].legend()
axs[0].text(0.5, -0.27, '(a) ImageNet-1K-1shot Benchmark', ha='center', va='center', transform=axs[0].transAxes, fontsize=15)

# Subplot (b)
for i, (x, y) in enumerate(zip(fpr95_b, auroc_b)):
    axs[1].scatter(x, y, marker=markers[i], label=methods[i],zorder=2, color=colors[i], s=sizes[i])
axs[1].set_xlabel('FPR95(%) ↓', fontsize=12)
axs[1].set_ylabel('AUROC(%) ↑', fontsize=13)

# Add legend and title below FPR95 for subplot (b)
axs[1].legend()
axs[1].text(0.5, -0.27, '(b) ImageNet-1K-16shot Benchmark', ha='center', va='center', transform=axs[1].transAxes, fontsize=15)

# 调整子图之间的间距
fig.subplots_adjust(top=0.97, bottom=0.3, left=0.1, right=0.9)
plt.subplots_adjust(wspace=0.1, hspace=0.38)  # 主要关注 wspace 参数
# plt.tight_layout()
plt.show()
plt.savefig('./mycaches_id/draw_data/F1.pdf')
