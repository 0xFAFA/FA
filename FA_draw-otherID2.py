import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi = 600)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
ax1.bar([1,     1.5,    2.0,    2.5,    3.0,    3.7], 
        [99.81, 99.95,  98.54,  99.40,  98.63,  99.88], 0.2, label="x", color = '#30ACC7', zorder=1,
 yerr = [0.02,  0.01,   0.19,   0.27,   0.23,   0.05], 
        capsize=5)
# ax2 = ax1.twinx()
ax1.bar([1.2,   1.7,    2.2,    2.7,    3.2,    3.9], 
        [99.19, 99.90,  97.49,  94.33,  97.67,  99.63], 0.2, label="x", color = "#ecaf98", zorder=1,
yerr =  [0.06,  0.02,   0.31,   0.81,   0.33,   0.08], 
        capsize=5)
ax1.set_ylabel("AUROC(%) ↑", fontsize = 15,labelpad=-1)
ax1.set_ylim(93.5, 100.001)
ax1.grid(True, alpha = 0.3)
ax1.legend([r'$\mathrm{FA}_{\mathrm{MCM}}$', r'$\mathrm{CoOp}_{\mathrm{MCM}}$'], fontsize =13)
ax1.set_xticks([1.1, 1.6, 2.1, 2.6, 3.1, 3.8], 
               ["Food\n101", "Stanford\nCars", 'Caltech\n101', 'FGVC\n Aircraft', 'Flowers\n102', "Oxford\nPets"],
               fontsize = 15)
ax1.tick_params(axis='y', labelcolor='black', labelsize=15)

plt.subplots_adjust(top=0.88, bottom=0.20, left=0.10, right=0.9)
plt.savefig('./mycaches_id/draw_data/F-otherID.pdf', ) #bbox_inches='tight'
# plt.show()