import matplotlib.pyplot as plt

for task in ["Math", "Code", "IF"]:
    methods = ["FF", "LoRA", "PiSSA", "JANUS-F", "JANUS-L", "JANUS-P"]
    # math
    ap1_results = {
        "Math": [39.97, 65.87, 67.55, 85.55, 106.43, 106.16],
        "Code": [73.57, 76.03, 79.71, 116.33, 105.52, 109.83],
        "IF": [50.26, 75.92, 79.53, 76.24, 105.08, 100.27]
    }
    ap2_results = {
        "Math": [100, 52.76, 72.56, 97.69, 53.91, 71.54],
        "Code": [100, 68.27, 75.94, 98.81, 67.76, 75.19],
        "IF": [100, 76.10, 81.80, 81.36, 74.78, 75.58],
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
        if (method == "JANUS-P" and task == "IF") or (method == "PiSSA" and task == "IF"):
            xytext = (0, 43)
        elif (method == "JANUS-F" and task == "IF"):
            xytext = (-105, -10)
        else:
            xytext = (0, -45)
        ax.annotate(
            method,
            (x, y),
            xytext=xytext,
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
            zorder=5,
        )


    # ax.set_title(f"Recovery in Stability  (LLaMA-2-7b on {task})", fontsize=30, fontweight="bold", pad=16)
    ax.set_xlabel("Stability measured by $\\text{AP}_1$ (%)", fontsize=30, fontweight="semibold")
    ax.set_ylabel("Plasticity measured by $\\text{AP}_2$ (%)", fontsize=30, fontweight="semibold")
    ax.title.set_color("#16202D")
    ax.xaxis.label.set_color("#243447")
    ax.yaxis.label.set_color("#243447")
    ax.tick_params(axis="both", labelsize=35, colors="#31465F")
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.1)
    if task == "Math":
        ax.set_xlim(36, 115)
        ax.set_ylim(40, 110)
    elif task == "Code":
        ax.set_xlim(70.5, 122)
        ax.set_ylim(60, 110)
    elif task == "IF":
        ax.set_xlim(48, 112)
        ax.set_ylim(55, 115)

    # plt.tight_layout()
    plt.savefig(f"/home/zhengzhilong/repo/peft/scripts/llama-2-7b-{task}-janus_multirun.pdf", dpi=300, facecolor=fig.get_facecolor())
