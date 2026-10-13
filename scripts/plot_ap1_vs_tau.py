import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

methods = ["JANUS-P (w/o) Math", "JANUS-P (w/o) Code", "JANUS-P (w/o) IF", "JANUS-L (w/o) Math", "JANUS-L (w/o) Code", "JANUS-L (w/o) IF", "JANUS-F (w/o) Math", "JANUS-F (w/o) Code", "JANUS-F (w/o) IF"]
tau = [0.7, 0.8, 0.9, 0.925, 0.95,]
# ap1 = [57.71, 74.43, 82.91, 84.94, 85.88,]  # 7b math ff
# ap1 = [98.74, 99.28, 99.28, 108.62, 108.45]  # 7b code pissa
# ap1 = [54.72, 77.01, 84.67, 84.67, 84.25]  # 8b math ff
ap1 = [81.66, 81.66, 94.26, 94.26, 94.01]  # 8b if lora
# ap1 = [54.72, 77.01, 84.67, 84.67, 84.25]  # 8b math ff
# tau = [0.7, 0.8, 0.9, 0.925, 0.95, 0.975]
# ap1 = [57.71, 74.43, 82.91, 84.94, 85.88, 82.03]



plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(18, 9))
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.90)
ax.set_position([0.12, 0.12, 0.86, 0.80])
ax.set_box_aspect(9 / 18)
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

for spine in ax.spines.values():
    spine.set_color("#000000")
    spine.set_linewidth(1.4)


ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="#9AA5BF")

ax.plot(tau, ap1, marker="o", markersize=12, color="#1F77B4", linewidth=3.0, alpha=0.9, zorder=3)
ax.set_xlabel("Acceptance ratio $\\tau$", fontsize=30, fontweight="semibold")
ax.set_ylabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
ax.title.set_color("#16202D")
ax.xaxis.label.set_color("#243447")
ax.yaxis.label.set_color("#243447")
ax.tick_params(axis="both", labelsize=30, colors="#31465F")
ax.set_axisbelow(True)
ax.margins(x=0.08, y=0.1)

# plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b_code_pissa-ap1-vs-tau.png", dpi=300, facecolor=fig.get_facecolor())
# plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b_code_pissa-ap1-vs-tau.pdf", dpi=300, facecolor=fig.get_facecolor())
plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-3-8b_if_lora-ap1-vs-tau.png", dpi=300, facecolor=fig.get_facecolor())
plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-3-8b_if_lora-ap1-vs-tau.pdf", dpi=300, facecolor=fig.get_facecolor())