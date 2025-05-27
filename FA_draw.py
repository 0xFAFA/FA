import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# 假设这是mcm和gl_mcm的数据，格式与原数据相同
FPR95_values_mcm = [
    [37.89, 29.60, 31.47, 30.78, 31.50],  # ImageNet1K
    [25.41, 7.96, 2.46, 0.48, 3.38],  # FGVC-Aircraft
    [31.21, 4.53, 2.42, 2.05, 1.43],  # UCF101
    [95.35, 58.09, 36.52, 22.78, 16.09]  # EuroSAT
]
AUROC_values_mcm = [
    [91.52, 93.72, 93.34, 93.51, 93.41],  # ImageNet1K
    [94.33, 98.34, 99.40, 99.79, 99.14],  # FGVC-Aircraft
    [93.91, 99.00, 99.51, 99.61, 99.72],  # UCF101
    [48.25, 82.19, 92.87, 95.93, 97.52]  # EuroSAT
]
ID_ACC_values_mcm = [
    [71.07, 70.96, 70.81, 70.86, 70.79],  # ImageNet1K
    [34.52, 34.03, 31.66, 33.61, 32.72],  # FGVC-Aircraft
    [77.99, 77.75, 78.29, 77.09, 78.11],  # UCF101
    [82.11, 81.92, 79.87, 78.21, 78.68]  # EuroSAT
]
# -----------
FPR95_values_gl_mcm = [
    [31.81, 25.69, 26.88, 26.73, 26.84],  # ImageNet1K
    [51.44, 17.95, 7.77, 2.82, 7.60],  # FGVC-Aircraft
    [26.77, 7.65, 4.45, 3.85, 3.17],  # UCF101
    [83.62, 64, 44.45, 30.68, 18.99]   # EuroSAT
]
AUROC_values_gl_mcm = [
    [91.51, 93.76, 93.62, 93.69, 93.65],  # ImageNet1K
    [86.37, 95.92, 98.27, 99.33, 98.11],  # FGVC-Aircraft
    [94.45, 98.34, 99.07, 99.21, 99.39],  # UCF101
    [66.77, 82.73, 92.55, 95.09, 96.74]   # EuroSAT
]
ID_ACC_values_gl_mcm = [
    [71.07, 70.96, 70.81, 70.86, 70.79],  # ImageNet1K
    [34.52, 34.03, 31.66, 33.61, 32.72],  # FGVC-Aircraft
    [77.99, 77.75, 78.29, 77.09, 78.11],  # UCF101
    [82.11, 81.92, 79.87, 78.21, 78.68]   # EuroSAT
]

colors = ['#00b6ed', '#d62728','black']
k_values = [1, 3, 5, 7, 9]
datasets = ['(a) ImageNet-1K', '(b) FGVCAircraft', '(c) UCF101', '(d) EuroSAT']
# 12个位置
legend_loc = ['upper right', 'lower right', 'upper right',
'upper right', 'lower right', 'upper right',
'upper right', 'lower right', 'upper right',
'upper right', 'lower right', 'upper right']


fig, axes = plt.subplots(3, 4, figsize=(20, 15))


for col in range(4):
    for row in range(3):
        ax = axes[row, col]
        if row == 0:  # FPR95
            y_values_mcm = FPR95_values_mcm[col]
            y_values_gl_mcm = FPR95_values_gl_mcm[col]
            ylabel = 'FPR95(%) ↓'
        elif row == 1:  # AUROC
            y_values_mcm = AUROC_values_mcm[col]
            y_values_gl_mcm = AUROC_values_gl_mcm[col]
            ylabel = 'AUROC(%) ↑'
        else:  # ID ACC
            y_values_mcm = ID_ACC_values_mcm[col]
            y_values_gl_mcm = ID_ACC_values_gl_mcm[col]
            ylabel = 'ID ACC(%) ↑'
        
        ax.plot(k_values, y_values_mcm, marker='D', linestyle=':', zorder=2, color=colors[row], label='MCM')
        ax.plot(k_values, y_values_gl_mcm, marker='o', linestyle='-', zorder=2, color=colors[row], label='GL_MCM')

        ax.set_xticks(k_values)
        ax.set_xlabel('The value of K', labelpad=1, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True)
        [grid.set_alpha(0.3) for grid in ax.get_xgridlines() + ax.get_ygridlines()]

        if row == 0 and col == 0:
            ax.set_ylim([24, 41])
        if row == 0 and col == 1:
            ax.set_ylim([-3, 55])
        if row == 2 and col == 0:
            ax.set_ylim([69, 73])
        if row == 2 and col == 1:
            ax.set_ylim([28, 39])
        if row == 2 and col == 2:
            ax.set_ylim([75, 81])
        if row == 2 and col == 3:
            ax.set_ylim([75, 85])
        if row == 1 and col == 3:
            ax.set_ylim([45, 100])
        if row == 0 and col == 3:
            ax.set_ylim([10, 100])

        cur_legend_loc = legend_loc[col * 3 + row]
        ax.legend(loc=cur_legend_loc, fontsize=12)
        if row == 2:  # 在每列的最后一行添加数据集标志
            ax.set_title(datasets[col], y=-0.44, fontsize=17)




fig.subplots_adjust(top=0.58, bottom=0.1, left=0.1, right=0.9)
plt.subplots_adjust(wspace=0.44, hspace=0.22)
plt.show()

plt.savefig('./mycaches_id/draw_data/F4.pdf')

