import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)

# 定义组数和每组点数
num_groups = 6
points_per_group = 1

# 定义形状和颜色
markers = [ '<', 'o']  
mks= ['o', 's', '^', '*', 'v','p', 'h', 'D']
colors = ['red', '#367dbd', 'orange', 'black','pink','purple']  
sizes = [200, 170] 

methods = [  r'$\mathrm{FA}_{\mathrm{MCM}}$'] # r'$\mathrm{CoOp}_{\mathrm{MCM}}$',

datasets=[ 'Food101', 'StandfordCars', 'Caltech101', 'Flowers102', 'OxfordPets', 'FGVCAircraft' ]

# 生成虚拟数据
# base_fpr = np.array([30.0, 32.5, 35.0, 37.5])  # 公共FPR坐标
# data = [
#     ([2.83, 0.61], [99.19, 99.81]),
#     ([0.09,  0.03], [99.93, 99.95]),
#     ([10.72,  6.35], [97.49, 98.54]),
#     ([9.52,  5.93], [97.62,  98.63]),
#     ([1.69,  0.39], [99.63, 99.88]),

#     ([25.1,  2.46], [94.5, 99.40])
# ]

# data = [
#     ([ 0.61], [ 99.81]),
#     ([  0.03], [ 99.95]),
#     ([  6.35], [ 98.54]),
#     ([ 5.93], [  98.63]),
#     ([ 0.39], [ 99.88]),

#     ([  2.46], [ 99.40])
# ]

data = [
    ([ 99.19], [ 99.81]),
    ([ 99.93], [ 99.95]),
    ([ 97.49], [ 98.54]),
    ([ 97.47], [  98.63]),
    ([ 99.63], [ 99.88]),

    ([ 94.33], [ 99.40])
]

# data = [
#     ([ 99.81], [ 99.19]),
#     ([ 99.95], [ 99.93]),
#     ([ 98.54], [ 97.49]),
#     ( [98.63], [ 97.47]),
#     ([ 99.88], [ 99.63]),

#     ([ 99.40], [ 94.33])
# ]

# for group_idx in range(num_groups):
#     # 生成AUROC值（基础值从92递减，加入随机扰动）
#     base_auroc = 92.0 - group_idx * 0.3
#     aurocs = base_auroc - 0.2 * np.arange(points_per_group) 
#     aurocs += np.random.normal(0, 0.1, points_per_group)
#     data.append((base_fpr, aurocs))

# print(data)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制每个组的数据点
for group_idx in range(num_groups):
    fprs, aurocs = data[group_idx]
    
    for point_idx in range(points_per_group):
        if group_idx == 1 and point_idx == 0:
            plt.scatter(
                x=fprs[point_idx],
                y=aurocs[point_idx],
                marker=mks[group_idx],
                color=colors[group_idx],
                s=sizes[point_idx],  # 点的大小
                edgecolors='black',  # 增加黑色边框
                linewidths=0.5,
                # label=f'{datasets[group_idx]} , {methods[point_idx]}',  # 为每个点设置唯一标签
                label=f'{datasets[group_idx]}',  # 为每个点设置唯一标签
                zorder=10
            )
        else:
            plt.scatter(
                x=fprs[point_idx],
                y=aurocs[point_idx],
                marker=mks[group_idx],
                color=colors[group_idx],
                s=sizes[point_idx],  # 点的大小
                edgecolors='black',  # 增加黑色边框
                linewidths=0.5,
                # label=f'{datasets[group_idx]} , {methods[point_idx]}'  # 为每个点设置唯一标签
                label=f'{datasets[group_idx]}'  # 为每个点设置唯一标签
            )

# 设置坐标轴范围和标签
# plt.xlim(-5, 30)
# plt.ylim(89.8, 92.2)  # 稍微扩展y轴范围
# plt.xlabel('FPR95(%) ↓', fontsize=18.5, labelpad=1)
plt.xlabel(r'AUROC(%)↑ | $\mathrm{CoOp}_{\mathrm{MCM}}$', fontsize=15.5, labelpad=1)
plt.ylabel(r'AUROC(%)↑ | $\mathrm{FA}_{\mathrm{MCM}}$', fontsize=15.5, labelpad=1)

# 设置网格和刻度
plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(np.arange(27.5, 41, 2.5))
# plt.yticks(np.arange(90, 92.5, 0.5))

# 添加图例，分成多列并竖着分布
plt.legend(
    bbox_to_anchor=(1.05, 1),  # 将图例放置在图形右侧
    loc='upper left',  # 图例的锚点位置
    borderaxespad=0.,  # 图例与图形的间距
    # fontsize=14.5,  # 图例字体大小
    # title="Legend",  # 图例标题
    # title_fontsize=13,  # 图例标题字体大小
    ncol=1,  # 将图例分成单列
    prop={ 'size': 20.6}
)

# 放大坐标轴刻度字体大小
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=15)  # 设置y轴刻度字体大小

# 调整布局
plt.subplots_adjust(top=0.98, bottom=0.650, left=0.20, right=0.45)
# plt.tight_layout()
plt.savefig('./mycaches_id/draw_data/F-otherID.pdf', ) #bbox_inches='tight'
plt.show()