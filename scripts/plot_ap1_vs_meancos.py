import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

methods = ["JANUS-P (w/o) Math", "JANUS-P (w/o) Code", "JANUS-P (w/o) IF", "JANUS-L (w/o) Math", "JANUS-L (w/o) Code", "JANUS-L (w/o) IF", "JANUS-F (w/o) Math", "JANUS-F (w/o) Code", "JANUS-F (w/o) IF"]
ap1 = [91.73, 98.93, 87.67, 88.74, 94.25, 89.29, 57.57, 104.9, 59.43]
meancos = [0.8630, 0.9002, 0.8611, 0.8540, 0.8911, 0.8655, 0.7783, 0.8065, 0.7722]

ap1 = [55.08, 83.76, 73.87, 69.29, 79.89, 81.65, 54.99, 86.88, 59.61]
meancos = [0.7337, 0.8285, 0.784, 0.8143, 0.9087, 0.8632, 0.7462, 0.922, 0.759]



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

point_count = min(len(methods), len(meancos), len(ap1))
fit_x = np.array(meancos[:point_count], dtype=float)
fit_y = np.array(ap1[:point_count], dtype=float)

if point_count >= 2:
    linreg = stats.linregress(fit_x, fit_y)
    slope = linreg.slope
    intercept = linreg.intercept
    fit_line_x = np.linspace(fit_x.min(), fit_x.max(), 200)
    fit_line_y = slope * fit_line_x + intercept
    ax.plot(
        fit_line_x,
        fit_line_y,
        color="#E45756",
        linewidth=3.0,
        alpha=0.9,
        zorder=2,
    )
    ax.text(
        0.98,
        0.02,
        f"p = {linreg.pvalue:.3g}",
        transform=ax.transAxes,
        fontsize=60,
        color="#000000",
        ha="right",
        va="bottom",
    )

offsets = [
    (0, 20), (-100, 0), (0, -20), (-100, 0),
    (0, -20), (90, 0), (0, -20), (0, 20), (90, 0),
]
offsets = [
    (25, 20), (0, 20), (0, -20), (0, -20),
    (0, 20), (0, 20), (0, -20), (-100, 0), (90, 0),
]
for (method, x, y), (dx, dy) in zip(zip(methods[:point_count], fit_x, fit_y), offsets):
    ax.scatter(x, y)
    ax.annotate(
        method,
        (x, y),
        xytext=(dx, dy),
        ha="center",
        va="center",
        textcoords="offset points",
        fontsize=20,
    )

ax.set_xlabel("JANUS shift measured by $\\bar{c}$", fontsize=30, fontweight="semibold")
ax.set_ylabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
ax.title.set_color("#16202D")
ax.xaxis.label.set_color("#243447")
ax.yaxis.label.set_color("#243447")
ax.tick_params(axis="both", labelsize=30, colors="#31465F")
ax.set_axisbelow(True)
ax.margins(x=0.08, y=0.1)

# plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b-ap1-vs-meancos.png", dpi=300, facecolor=fig.get_facecolor())
# plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b-ap1-vs-meancos.pdf", dpi=300, facecolor=fig.get_facecolor())
plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-3-8b-ap1-vs-meancos.png", dpi=300, facecolor=fig.get_facecolor())
plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-3-8b-ap1-vs-meancos.pdf", dpi=300, facecolor=fig.get_facecolor())
