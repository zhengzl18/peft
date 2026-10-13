import matplotlib.pyplot as plt

for task in ["Math", "Code", "IF"]:
    methods = ["LoRA", "PiSSA", "CorDA", "MiLoRA", "LoRA-Null", "JANUS-P"]
    # methods = ["FF", "LoRA", "PiSSA", "CorDA", "MiLoRA", "LoRA-Null", "FF JANUS", "LoRA JANUS", "PiSSA JANUS"]
    # math
    ap1_results = {
        "Math": [65.87, 67.55, 80.05, 75.02, 80.44, 106.16],
        "Code": [76.03, 79.71, 85.28, 77.50, 88.75, 109.83],
        "IF": [75.92, 79.53, 98.46, 84.75, 92.51, 100.27]
        # "Math": [39.97, 65.87, 71.27, 80.05, 75.02, 80.44, 111.55],
        # "Code": [73.57, 76.03, 79.52, 85.28, 77.50, 88.75, 108.45],
        # "IF": [50.26, 75.92, 79.47, 98.46, 84.75, 92.51, 99.05]
    }
    ap2_results = {
        "Math": [52.76, 72.56, 61.12, 53.82, 58.06, 71.54],
        "Code": [68.27, 75.94, 66.93, 62.35, 68.92, 75.19],
        "IF": [76.10, 81.80, 72.37, 60.96, 78.07, 75.58],
        # "Math": [100, 52.76, 73.96, 61.12, 53.82, 58.06, 97.69, 53.91, 73.01],
        # "Code": [100, 68.27, 76.83, 66.93, 62.35, 68.92, 98.81, 67.76, 76.05],
        # "IF": [100, 76.10, 95.39, 72.37, 60.96, 78.07, 81.36, 74.78, 82.02],
    }

    ap1 = ap1_results[task]
    ap2 = ap2_results[task]


    point_styles = {
        # "FF": {"color": "#4C78A8", "marker": "o", "size": 490},
        "LoRA": {"color": "#4C78A8", "marker": "o", "size": 1800},
        "PiSSA": {"color": "#15EDCC", "marker": "D", "size": 1500},
        "CorDA": {"color": "#54A24B", "marker": "s", "size": 1800},
        "MiLoRA": {"color": "#E45756", "marker": "^", "size": 1800},
        "LoRA-Null": {"color": "#72B7B2", "marker": "P", "size": 2100},
        # "FF JANUS": {"color": "#FDB600", "marker": "*", "size": 1500},
        # "LoRA JANUS": {"color": "#FDB600", "marker": "*", "size": 1500},
        "JANUS-P": {"color": "#FDB600", "marker": "*", "size": 3000},
    }

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(13.5, 9))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.90)
    ax.set_position([0.12, 0.12, 0.86, 0.80])
    ax.set_box_aspect(9 / 15)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.4)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="#9AA5BF")


    for method, x, y in zip(methods, ap1, ap2):
        style = point_styles[method]
        ax.scatter(
            x,
            y,
            s=style["size"],
            marker=style["marker"],
            c=style["color"],
            edgecolors="#F5F7FB" if "JANUS" not in method  else "#1F2A38",
            linewidths=2.0 if "JANUS" in method else 1.2,
            alpha=0.98 if "JANUS" in method else 0.9,
            zorder=4 if "JANUS" in method else 3,
        )
        ax.annotate(
            method,
            (x, y),
            xytext=(0, 40) if (method == "CorDA" and task == "Math") or (method == "LoRA-Null" and task == "Code") or (method == "JANUS-P" and task == "IF") else (0, -45),
            ha="center",
            va="center",
            textcoords="offset points",
            fontsize=35,
            fontweight="bold" if "JANUS" in method else "medium",
            color="#FFA200FF" if "JANUS" in method else "#1F2D3D",
            zorder=5 if "JANUS" in method else 4,
        )


    # ax.set_title(f"Stability v.s. Plasticity (LLaMA-2-7b on {task})", fontsize=30, fontweight="bold", pad=16)
    ax.set_xlabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
    ax.set_ylabel("Plasticity measured by $\\text{AP}_2$ (%)", fontsize=30, fontweight="semibold")
    ax.title.set_color("#16202D")
    ax.xaxis.label.set_color("#243447")
    ax.yaxis.label.set_color("#243447")
    ax.tick_params(axis="both", labelsize=35, colors="#31465F")
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.1)

    if task == "Math":
        ax.set_xlim(60, 112)
        ax.set_ylim(48, 77)
    elif task == "Code":
        ax.set_xlim(73, 115)
        ax.set_ylim(58, 80)
    elif task == "IF":
        ax.set_xlim(73, 103.5)
        ax.set_ylim(45, 105)

    # plt.tight_layout()
    plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b-{task}_multirun.pdf", dpi=300, facecolor=fig.get_facecolor())
    plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b-{task}_multirun.png", dpi=300, facecolor=fig.get_facecolor())