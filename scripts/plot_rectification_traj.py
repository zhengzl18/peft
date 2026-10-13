import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats
from matplotlib.patches import ConnectionPatch

step2color = {
    0: "#E8831F",
    1: "#DBC130",
    2: "#64B95C",
} 
points = [
    {
        "label": "$\hat{\\theta}$",
        "step": "fine-tuned",
        "ap1": 39.97,
        "ap2": 100,
        "offset": (0, -30),
        "style":{
            "color": "#FA8C8C",
        }
    },
    {
        "label": "$\\alpha_0=1$\n$\\bar{c}=0.746$",
        "step": "Step 0",
        "ap1": 54.67,
        "ap2": 100.43,
        "offset": (0, -40),
        "style":{
            "facecolors": mcolors.to_rgba(step2color[0], alpha=0.3),
            "edgecolors": step2color[0],
            "linewidths": 2,
            "linestyle": '--'
        }
    },
    {
        "label": "$\\alpha_0=\\beta$\n$\\bar{c}=0.840$",
        "step": "Step 0",
        "ap1": 66.79,
        "ap2": 98.17,
        "offset": (0, -40),
        "style":{
            "facecolors": mcolors.to_rgba(step2color[0], alpha=0.3),
            "edgecolors": step2color[0],
            "linewidths": 2,
            "linestyle": '--'
        }
    },
    {
        "label": "$\\alpha_0=\\beta^2$\n$\\bar{c}=0.900$",
        "step": "Step 0",
        "ap1": 83.46,
        "ap2": 96.40,
        "offset": (0, -50),
        "style":{
            "facecolors": mcolors.to_rgba(step2color[0], alpha=0.3),
            "edgecolors": step2color[0],
            "linewidths": 2,
            "linestyle": '--'
        }
    },
    {
        "label": "$\\alpha_0=\\beta^3$\n$\\bar{c}=0.944$",
        "step": "Step 0",
        "ap1": 88.77,
        "ap2": 89.91,
        "offset": (0, -40),
        "style":{
            "facecolors": mcolors.to_rgba(step2color[0], alpha=0.3),
            "edgecolors": step2color[0],
            "linewidths": 2,
            "linestyle": '--'
        }
    },
    {
        "label": "$\\alpha_0=\\beta^4$\n$\\bar{c}=0.961$",
        "step": "Step 0 final",
        "ap1": 93.95,
        "ap2": 52.16,
        "offset": (0, -40),
        "style":{
            "color": step2color[0],
        }
    },
    {
        "label": "$\\alpha_1=1$",
        "step": "Step 1",
        "ap1": 76.53,
        "ap2": 100.27,
        "offset": (0, -30),
        "style":{
            "facecolors": mcolors.to_rgba(step2color[1], alpha=0.3),
            "edgecolors": step2color[1],
            "linewidths": 2,
            "linestyle": '--'
        }
    },
    {
        "label": "$\\alpha_1=\\beta$",
        "step": "Step 1",
        "ap1": 77.70,
        "ap2": 99.82,
        "offset": (0, -30),
        "style":{
            "facecolors": mcolors.to_rgba(step2color[1], alpha=0.3),
            "edgecolors": step2color[1],
            "linewidths": 2,
            "linestyle": '--'
        }
    },
    {
        "label": "$\\alpha_1=\\beta^2$\n$\\bar{c}=0.969$",
        "step": "Step 1 final",
        "ap1": 93.27,
        "ap2": 98.08,
        "offset": (0, -50),
        "style":{
            "color": step2color[1],
        }
    },
    {
        "label": "$\\alpha_2=1$\n$\\bar{c}=0.988$",
        "step": "Step 2 final",
        "ap1": 91.56,
        "ap2": 101.00,
        "offset": (0, 50),
        "style":{
            "color": step2color[2],
        }
    },
    {
        "label": "$\\theta_0$",
        "step": "pretrain",
        "ap1": 100,
        "ap2": 59.72,
        "offset": (0, -30),
        "style":{
            "color": "#BABABA",
        }
    },
]

plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(15, 12))
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.90)
ax.set_position([0.12, 0.12, 0.86, 0.80])
ax.set_box_aspect(12 / 15)
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

for spine in ax.spines.values():
    spine.set_color("#000000")
    spine.set_linewidth(1.4)

ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="#9AA5BF")

# ================= 局部放大镜设置 =================
# 在左下角空白处放置放大镜窗口 (参数: [左, 下, 宽, 高] 的相对比例)
axins = ax.inset_axes([0.12, 0.08, 0.6, 0.47])
axins.set_xlim(75, 96)     # 聚焦 X 轴范围
axins.set_ylim(93, 104.25)  # 聚焦 Y 轴范围
axins.set_facecolor("#FFFFFF")
axins.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="#9AA5BF")
for spine in axins.spines.values():
    spine.set_color("#000000")
    spine.set_linewidth(1.4)
axins.tick_params(axis="both", labelsize=20, colors="#31465F")

# 判断点是否在需要放大的密集区域内
def in_zoom_area(x, y):
    return 74 <= x <= 96 and 90 <= y <= 103
# =================================================


for point in points:
    # 决定当前的标注应该画在主图还是放大镜图里
    target_ax = axins if in_zoom_area(point["ap1"], point["ap2"]) else ax
    
    # 散点在两张图里都画出来
    ax.scatter(point["ap1"], point["ap2"], **point["style"], s=1100, zorder=4)
    axins.scatter(point["ap1"], point["ap2"], **point["style"], s=2500, zorder=4)
    
    # 只在对应的图中添加文字标注
    if "Step 0 final" in point["step"]:
        target_ax.annotate("$\\theta_1$", (point["ap1"], point["ap2"]), ha="center", va="center", xytext=(0, -1), textcoords="offset points", fontsize=23, zorder=5)
    elif "Step 1 final" in point["step"]:
        target_ax.annotate("$\\theta_2$", (point["ap1"], point["ap2"]), ha="center", va="center", xytext=(0, -1), textcoords="offset points", fontsize=25, zorder=5)
    elif "Step 2 final" in point["step"]:
        target_ax.annotate("$\\theta_\\text{rect}$", (point["ap1"], point["ap2"]), ha="center", va="center", fontsize=24, zorder=5)
    elif "fine-tuned" in point["step"]:
        target_ax.annotate("$\hat{\\theta}$", (point["ap1"], point["ap2"]), ha="center", va="center", xytext=(0, -5), textcoords="offset points", fontsize=22, zorder=5)
    elif "pretrain" in point["step"]:
        target_ax.annotate("$\\theta_0$", (point["ap1"], point["ap2"]), ha="center", va="center", fontsize=23, zorder=5)
        
    if "Step " in point["step"] and point["step"] != "Step 1":
        target_ax.annotate(point["label"], (point["ap1"], point["ap2"]), xytext=point['offset'], ha="center", va="center", textcoords="offset points", fontsize=23, zorder=5)

# 手动标注带有箭头的文本（由于都在右上角，直接移入 axins）
# 为了避免超出 axins 边界，略微调整了 xytext 的偏移量
axins.annotate(
    "$\\alpha_1=1$\n$\\bar{c}=0.922$",
    (76.53, 101),
    xytext=(20, 50), 
    ha="center",
    va="center",
    textcoords="offset points",
    arrowprops=dict(facecolor='black', lw=1, arrowstyle="->"),
    fontsize=23,
    zorder=5,
)
axins.annotate(
    "$\\alpha_1=\\beta$\n$\\bar{c}=0.952$",
    (77.7, 99),
    xytext=(0, -50),
    ha="center",
    va="center",
    textcoords="offset points",
    arrowprops=dict(facecolor='black', lw=1, arrowstyle="->"),
    fontsize=23,
    zorder=5,
)

for i in range(4):
    step_i_points = [point for point in points if f"Step {i}" in point["step"]]
    step_i_minus1_points = [point for point in points if f"Step {i-1}" in point["step"]]
    if step_i_points:
        if step_i_minus1_points:
            ap1s = [points[0]["ap1"],] + [point["ap1"] for point in step_i_points] + [step_i_minus1_points[-1]["ap1"],]
            ap2s = [points[0]["ap2"],] + [point["ap2"] for point in step_i_points] + [step_i_minus1_points[-1]["ap2"],]
        else:
            ap1s = [points[0]["ap1"],] + [point["ap1"] for point in step_i_points] + [points[-1]["ap1"],]
            ap2s = [points[0]["ap2"],] + [point["ap2"] for point in step_i_points] + [points[-1]["ap2"],]
        
        # 在主图和放大镜图中都画上连线
        ax.plot(ap1s, ap2s, color=step2color[i], linewidth=4, alpha=0.3, zorder=2, linestyle="--")
        ax.plot(ap1s[-2:], ap2s[-2:], color=step2color[i], linewidth=4, alpha=1, zorder=3)
        axins.plot(ap1s, ap2s, color=step2color[i], linewidth=4, alpha=0.3, zorder=2, linestyle="--")
        axins.plot(ap1s[-2:], ap2s[-2:], color=step2color[i], linewidth=4, alpha=1, zorder=3)

# 坐标轴和样式设置
ax.set_xlabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
ax.set_ylabel("Plasticity measured by $\\text{AP}_2$ (%)", fontsize=30, fontweight="semibold")
ax.title.set_color("#16202D")
ax.xaxis.label.set_color("#243447")
ax.yaxis.label.set_color("#243447")
ax.tick_params(axis="both", labelsize=30, colors="#31465F")
ax.set_axisbelow(True)
ax.margins(x=0.08, y=0.1)

ax.set_xlim(38, 102)
ax.set_ylim(-10, 110)

# 背景阴影线（主图和放大图均添加，保持视觉一致）
for target in [ax, axins]:
    target.axhline(y=93, color="#2A86E2", linestyle='--', linewidth=2, zorder=1, alpha=0.2)
    target.axhspan(ymin=93, ymax=110, facecolor='#2A86E2', alpha=0.1, zorder=1)
    target.axvline(x=90, color="#2A86E2", linestyle='--', linewidth=2, zorder=1, alpha=0.2)
    target.axvspan(xmin=90, xmax=102, facecolor='#2A86E2', alpha=0.1, zorder=1)

ax.annotate(
    "HIGH PLASTICITY ",
    (70, 106.5),
    color="#2A86E2",
    ha="center",
    va="center",
    fontweight="bold",
    fontsize=30,
    alpha=0.5
)
ax.annotate(
    "HIGH STABILITY",
    (100.5, 50),
    color="#2A86E2",
    rotation=-90,
    ha="center",
    va="center",
    fontweight="bold",
    fontsize=30,
    alpha=0.5
)

# 绘制主图与放大镜之间的连接框线
rect, connects = ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2, alpha=0.6)

# rect, connects = ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2, alpha=0.6)
for c in connects:
    c.set_visible(False)
connects[2].set_visible(True)

# 2. 找到你需要连接的具体坐标
# 小框（放大镜）的左下角：在 axins 自己的相对坐标系中是 (0, 0)
# 大框（原图目标区域）的左上角：根据你设定的 xlim(74, 96) 和 ylim(97, 101.5)，左上角坐标是 (74, 101.5)

# 3. 创建自定义连线
con = ConnectionPatch(
    xyA=(0, 1), coordsA=axins.transAxes,    # 起点 A：小框的左下角 (使用相对坐标)
    xyB=(75, 93), coordsB=ax.transData,  # 终点 B：大框的左上角 (使用原图的数据坐标)
    color="black",        # 颜色与之前的框线保持一致
    linewidth=2,          # 线宽
    alpha=0.6,            # 透明度
    linestyle="-"         # 实线，如果是虚线可以用 "--"
)

# 4. 把这条线添加到主图中
ax.add_patch(con)

plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/rectification_traj.png", dpi=300, facecolor=fig.get_facecolor())
plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/rectification_traj.pdf", dpi=300, facecolor=fig.get_facecolor())