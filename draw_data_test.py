import matplotlib.pyplot as plt
import numpy as np


# 设置字体大小
plt.rcParams.update({'font.size': 12})

# 生成模拟数据
methods = ['CoOp', 'TaskRes', 'Plot', 'LoCoOp', 'GalLop', 'Ours']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 模拟数据
data = {
    'Average': {
        'CoOp': [0, 0, 0, 0, 0],
        'TaskRes': [0, 0, 0, 0, 0],
        'Plot': [0, 0, 0, 0, 0],
        'LoCoOp': [0, 0, 0, 0, 0],
        'GalLop': [0, 0, 0, 0, 0],
        'Ours': [0, 0, 0, 0, 0]
    },
    'SD198(42+156)': {
        'CoOp': [24.70, 25.18, 37.29, 42.76, 51.78],
        'TaskRes': [29.93, 36.10, 42.23, 50.12, 60.33],
        'Plot': [27.55, 31.35, 35.87, 46.79, 56.29],
        'LoCoOp': [25.90, 30.20, 31.40, 48.00, 52.00],
        'GalLop': [25.18, 32.54, 33.97, 45.84, 62.71],
        'Ours': [28.50, 35.39, 43.23, 51.31, 66.03]
    },
    'LC25000': {
        'CoOp': [56.38, 68.06, 78.34, 82.32, 88.14],
        'TaskRes': [65.24, 71.92, 81.52, 83.72, 88.92],
        'Plot': [65.60, 74.30, 79.26, 82.44, 88.80],
        'LoCoOp': [50.70, 70.70, 83.20, 85.70, 90.90],
        'GalLop': [54.16, 69.80, 75.36, 75.54, 80.80],
        'Ours': [62.36, 74.04, 86.56, 87.84, 93.74] 
    },
    'NCT-CRC-HE-100K': {
        'CoOp': [53.45, 56.39, 66.10, 68.13, 73.64],
        'TaskRes': [50.24, 57.45, 73.48, 74.36, 79.69],
        'Plot': [55.72, 58.80, 70.08, 77.52, 78.68],
        'LoCoOp': [47.80, 60.30, 63.00, 71.60, 76.70],
        'GalLop': [60.72, 66.66, 74.65, 78.30, 90.00],
        'Ours': [58.59, 67.02, 81.43, 80.82, 88.80]
    },

}

# 散点数据
scatter_data = {
    'Average': {'clip zero-shot': [(0, 21.62),  ]},
    'SD198(42+156)': {'clip zero-shot': [(0, 15.68),  ]},
    'LC25000': {'clip zero-shot': [(0, 31.28), ]},
    'NCT-CRC-HE-100K': {'clip zero-shot': [(0, 17.91), ]},

}
# 获取数据集的数量
num_datasets = len(data)

# 计算average
temp_average_data = {}
for method in methods:
    temp_average_data[method] = [0, 0, 0, 0, 0]


for dataset, dataset_data in data.items():
    if dataset == 'Average':
        continue
    for method in methods:
        for i in range(5):
            temp_average_data[method][i] += dataset_data[method][i]
        
for method in methods:
    for i in range(5):
        temp_average_data[method][i] /= (num_datasets - 1)
data['Average'] = temp_average_data

print(temp_average_data)

# 计算子图的行列数
num_rows = (num_datasets + 1) // 2  # 确保至少有一行
num_cols = min(num_datasets, 2)

# 创建一个包含多个子图的图表
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

# 定义 x 轴的标签
x_ticks = [0,1,2,4,8,16]
fewshot_ticks = [ 1, 2, 4, 8, 16]

# 绘制每个数据集的图表
for i, (dataset, dataset_data) in enumerate(data.items()):
    # 计算axes的横纵坐标
    index_x = i // 2
    index_y = i % 2

    for method, color in zip(methods, colors):
        data_list = dataset_data[method]
        axes[index_x][index_y].plot(fewshot_ticks, data_list, marker='o', label=method, color=color, zorder=1)

    # 先绘制网格线
    axes[index_x][index_y].grid(True)

    # 添加散点数据
    scatter_points = scatter_data.get(dataset, {}).get('clip zero-shot', [])
    for x, y in scatter_points:
        axes[index_x][index_y].scatter(x, y, color='black', marker='o', s=100, zorder=3)  # 画散点, s是点的大小,marker能够取值为'o', 'x', '+', 'v', '^', '<', '>', 's', 'd'等
        axes[index_x][index_y].annotate(f'clip zero-shot', (x, y), textcoords="offset points", xytext=(10,10), ha='center')  # 标注散点

    axes[index_x][index_y].set_xlabel('# shots per class')
    axes[index_x][index_y].set_ylabel('Top-1 Accuracy (%)')
    axes[index_x][index_y].legend(loc='lower right')
    axes[index_x][index_y].set_title(dataset)
    axes[index_x][index_y].set_xticks(x_ticks)  # 设置 x 轴的标签
 


# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# save
plt.savefig('./mycaches/draw_data/plot_test.png')