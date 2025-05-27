import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi = 600)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))

idx = 0
k               = [1,       2,      3,      4,      5,          6]
common_aurco    = [93.525,  93.757, 93.823, 93.88,  93.97,    93.888]
common_fpr      = [25.894,  25.692, 25.683, 25.94,  25.5,     25.98]
common_baseline_fpr = 27.27
common_baseline_aur = 93.31

challenge_auroc = [83.7, 83.955,    83.93,  83.75,  83.855, 83.57]
challenge_fpr   = [53.835,  53.595, 53.61,  54.275, 54.29,  54.42]
challenge_baseline_auroc = 82.24
challenge_baseline_fpr   = 57.40

idx = 0
ax1 = axs[idx]
ax2 = ax1.twinx()
ax1.plot(k, common_aurco, 'o-', color="#f4b183", linewidth=2, markersize=6)
ax2.plot(k, common_fpr, '^-', color='#00b6ed', label='FPR95', linewidth=2, markersize=6)

ax1.axhline(y=common_baseline_aur, color='#f4b183', linestyle='--', linewidth=3)
ax2.axhline(y=common_baseline_fpr, color='#00b6ed', linestyle='--', linewidth=3)

ax1.set_ylim(89, 96.001)
ax1.set_xticks(np.arange(1, 7, 1))
ax1.set_yticks(np.arange(89, 96.001, 2))
ax2.set_ylim(20, 50)
# ax2.set_yticks(np.arange(70, 95, 5))


# ax1.set_xlabel(r'K' , fontsize=20, )
ax1.set_ylabel(r'AUROC(%) ↑',  color="#f4b183", fontsize=14, fontweight='bold')
ax2.set_ylabel(r'FPR95(%) ↓',  color="#00b6ed", fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.legend(["AUROC"], loc='upper right', fontsize=13)#
ax2.legend(["FPR95"], loc='upper left', fontsize=13)#
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.tick_params(axis='x', labelcolor='black', labelsize=18)
ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.set_title('(a) Conventional OOD benchmark', y = -0.25, fontsize = 16)

idx = 1
ax1 = axs[idx]
ax2 = ax1.twinx()
ax1.plot(k, challenge_auroc, 'o-', color="#f4b183", linewidth=2, markersize=6)
ax2.plot(k, challenge_fpr, '^-', color='#00b6ed', label='FPR95', linewidth=2, markersize=6)

ax1.axhline(y=challenge_baseline_auroc, color='#f4b183', linestyle='--', linewidth=3)
ax2.axhline(y=challenge_baseline_fpr, color='#00b6ed', linestyle='--', linewidth=3)

ax1.set_ylim(70, 90.001)
ax1.set_xticks(np.arange(1, 7, 1))
# ax1.set_yticks(np.arange(75, 90.001, 5))
ax2.set_ylim(45, 85)
ax2.set_yticks(np.arange(45, 90, 10))


# ax1.set_xlabel(r'K' , fontsize=20, )
ax1.set_ylabel(r'AUROC(%) ↑',  color="#f4b183", fontsize=14, fontweight='bold')
ax2.set_ylabel(r'FPR95(%) ↓',  color="#00b6ed", fontsize=14, fontweight='bold')

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.legend(["AUROC"], loc='upper right', fontsize=13)#
ax1.set_title('(b) Challenging OOD benchmark', y = -0.25, fontsize = 16)
ax2.legend(["FPR95"], loc='upper left', fontsize=13)#
ax1.tick_params(axis='x', labelcolor='black', labelsize=18)
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,  wspace=0.65)
# plt.savefig("sensitive.pdf", format="pdf", bbox_inches="tight")
plt.savefig('./mycaches_id/draw_data/Fs.pdf', bbox_inches="tight")