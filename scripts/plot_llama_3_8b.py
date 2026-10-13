import matplotlib.pyplot as plt

for task in ["Math", "Code", "IF"]:
    methods = ["LoRA", "PiSSA", "CorDA", "MiLoRA", "LoRA-Null", "JANUS-L"]
    # methods = ["FF", "LoRA", "PiSSA", "CorDA", "MiLoRA", "LoRA-Null", "FF JANUS", "LoRA JANUS", "PiSSA JANUS"]
    # math
    ap1_results = {
        "Math": [57.02, 34.49, 66.96, 70.78, 66.22, 87.07],
        "Code": [71.17, 74.42, 77.52, 87.28, 75.99, 89.28],
        "IF": [75.96, 69.55, 73.80, 92.97, 67.77, 91.23],
        # "Math": [44.19, 55.52, 34.49, 66.96, 70.78, 66.22, 84.25, 90.13, 77.56],
        # "Code": [74.91, 72.16, 74.42, 77.52, 87.28, 75.99, 94.17, 89.35, 91.16],
        # "IF": [51.66, 77.03, 69.55, 73.80, 92.97, 67.77, 81.93, 94.01, 83.91],
    }
    ap2_results = {
        "Math": [100.73, 102.92, 102.67, 99.10, 103.65, 100.54],
        "Code": [110.46, 115.98, 110.51, 111.90, 113.11, 109.50],
        "IF": [101.85, 113.30, 98.99, 90.74, 100.34, 100.00],
        # "Math": [100.00, 100.94, 102.92, 102.67, 99.10, 103.65, 101.17, 100.51, 103.37],
        # "Code": [100.00, 111.00, 115.98, 110.51, 111.90, 113.11, 96.66, 110.59, 115.87],
        # "IF": [100.00, 99.83, 113.30, 98.99, 90.74, 100.34, 93.77, 98.82, 106.23],
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
        "JANUS-L": {"color": "#FDB600", "marker": "*", "size": 3000},
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
            xytext=(0, 40) if (method == "LoRA" and task == "IF") or (method == "LoRA-Null" and task == "Math") else (0, -45),
            ha="center",
            va="center",
            textcoords="offset points",
            fontsize=35,
            fontweight="bold" if "JANUS" in method else "medium",
            color="#FFA200FF" if "JANUS" in method else "#1F2D3D",
            zorder=5 if "JANUS" in method else 4,
        )


    # ax.set_title(f"Stability v.s. Plasticity (LLaMA-3-8b on {task})", fontsize=30, fontweight="bold", pad=16)
    ax.set_xlabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
    ax.set_ylabel("Plasticity measured by $\\text{AP}_2$ (%)", fontsize=30, fontweight="semibold")
    ax.title.set_color("#16202D")
    ax.xaxis.label.set_color("#243447")
    ax.yaxis.label.set_color("#243447")
    ax.tick_params(axis="both", labelsize=35, colors="#31465F")
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.1)
    if task == "Math":
        ax.set_xlim(30, 94)
        ax.set_ylim(91, 115)
    elif task == "Code":
        ax.set_xlim(70, 91.2)
        ax.set_ylim(95, 130)
    elif task == "IF":
        ax.set_xlim(63, 98)
        ax.set_ylim(80, 120)

    # plt.tight_layout()
    plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-3-8b-{task}_multirun.pdf", dpi=300, facecolor=fig.get_facecolor())