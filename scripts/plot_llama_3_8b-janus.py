import matplotlib.pyplot as plt

for task in ["Math", "Code", "IF"]:
    methods = ["FF", "LoRA", "PiSSA", "JANUS-F", "JANUS-L", "JANUS-P"]
    # math
    ap1_results = {
        "Math": [44.19, 57.02, 34.49, 84.25, 87.07, 77.56],
        "Code": [74.91, 71.17, 74.42, 94.17, 89.28, 91.16],
        "IF": [51.66, 75.96, 69.55, 81.93, 91.23, 83.91],
        # "Math": [44.19, 55.52, 34.49, 66.96, 70.78, 66.22, 84.25, 90.13, 77.56],
        # "Code": [74.91, 72.16, 74.42, 77.52, 87.28, 75.99, 94.17, 89.35, 91.16],
        # "IF": [51.66, 77.03, 69.55, 73.80, 92.97, 67.77, 81.93, 94.01, 83.91],
    }
    ap2_results = {
        "Math": [100.00, 100.73, 102.92, 101.17, 100.54, 103.37],
        "Code": [100.00, 110.46, 115.98, 96.66, 109.50, 115.87],
        "IF": [100.00, 101.85, 113.30, 93.77, 100.00, 106.23],
        # "Math": [100.00, 100.94, 102.92, 102.67, 99.10, 103.65, 101.17, 100.51, 103.37],
        # "Code": [100.00, 111.00, 115.98, 110.51, 111.90, 113.11, 96.66, 110.59, 115.87],
        # "IF": [100.00, 99.83, 113.30, 98.99, 90.74, 100.34, 93.77, 98.82, 106.23],
    }

    ap1 = ap1_results[task]
    ap2 = ap2_results[task]

    point_styles = {
        "FF": {"color": "#4C78A8", "marker": "o", "size": 1800},
        "LoRA": {"color": "#4C78A8", "marker": "*", "size": 3300},
        "PiSSA": {"color": "#4C78A8", "marker": "s", "size": 1700},
        "JANUS-F": {"color": "#FDB600", "marker": "o", "size": 1800},
        "JANUS-L": {"color": "#FDB600", "marker": "*", "size": 3000},
        "JANUS-P": {"color": "#FDB600", "marker": "s", "size": 1700},
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
            xytext=(0, 43) if (method == "JANUS-F" and task == "Math") else (0, -45),
            ha="center",
            va="center",
            textcoords="offset points",
            fontsize=35,
            fontweight="bold" if "JANUS" in method else "medium",
            color="#FFA200FF" if "JANUS" in method else "#1F2D3D",
        )

    points = dict(zip(methods, zip(ap1, ap2)))
    arrow_styles = {
        ("FF", "JANUS-F"): {"color": "#FFA200FF"},
        ("LoRA", "JANUS-L"): {"color": "#FFA200FF"},
        ("PiSSA", "JANUS-P"): {"color": "#FFA200FF"},
    }
    for src, dst in [("FF", "JANUS-F"), ("LoRA", "JANUS-L"), ("PiSSA", "JANUS-P")]:
        arrow_style = arrow_styles[(src, dst)]
        ax.annotate(
            "",
            xy=points[dst],
            xytext=points[src],
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_style["color"],
                lw=3.0,
                shrinkA=25,
                shrinkB=25,
                capstyle="round",
                joinstyle="round",
                mutation_scale=30,
            ),
            zorder=2,
        )


    # ax.set_title(f"Recovery in Stability (LLaMA-3-8b on {task})", fontsize=30, fontweight="bold", pad=16)
    ax.set_xlabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
    ax.set_ylabel("Plasticity measured by $\\text{AP}_2$ (%)", fontsize=30, fontweight="semibold")
    ax.title.set_color("#16202D")
    ax.xaxis.label.set_color("#243447")
    ax.yaxis.label.set_color("#243447")
    ax.tick_params(axis="both", labelsize=35, colors="#31465F")
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.1)
    if task == "Math":
        ax.set_xlim(28, 94)
        ax.set_ylim(99, 104)
    elif task == "Code":
        ax.set_xlim(69, 97)
        ax.set_ylim(90, 125)
    elif task == "IF":
        ax.set_xlim(49, 96)
        ax.set_ylim(80, 125)

    # plt.tight_layout()
    plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-3-8b-{task}-janus_multirun.pdf", dpi=300, facecolor=fig.get_facecolor())
