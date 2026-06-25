#!/usr/bin/env python3
"""
INFOCOM 2027 — Motivation Figures (2-Jump, clean 2+2 layout)
Fig 1: (a) GSM8K EM, (b) Dolly Gen Length + ROUGE-L
Fig 2: (a) Jaccard per round, (b) Union + Fully Shared modules
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False
    from PIL import Image, ImageDraw, ImageFont

if HAS_MPL:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_motivation")
os.makedirs(OUT, exist_ok=True)

C = {"red":"#d62728","orange":"#ff7f0e","blue":"#1f77b4","green":"#2ca02c",
     "gray":"#7f7f7f","purple":"#9467bd"}

DOLLY_DENSE_GENLEN = 51.6567
DOLLY_DENSE_ROUGEL = 35.1083


def fig2_data():
    """Shared data for all Fig. 2 variants."""
    rounds = np.array([1, 2, 3, 4, 5])
    return {
        "rounds": rounds,
        "methods": ["AB-Effective", "AB-Factor", "Ours"],
        "jac": {
            "AB-Effective": np.array([0.82, 1.00, 1.00, 1.00, 1.00]),
            "AB-Factor": np.array([0.74, 0.84, 0.60, 0.55, 0.43]),
            "Ours": np.array([0.19, 0.17, 0.18, 0.17, 0.16]),
        },
        "union": {
            "AB-Effective": np.array([24, 20, 20, 20, 20]),
            "AB-Factor": np.array([30, 34, 32, 44, 52]),
            "Ours": np.array([80, 80, 80, 84, 84]),
        },
        "shared": {
            "AB-Effective": np.array([10, 20, 20, 20, 20]),
            "AB-Factor": np.array([10, 12, 8, 6, 6]),
            "Ours": np.array([0, 0, 0, 0, 0]),
        },
        "colors": {
            "AB-Effective": C["red"],
            "AB-Factor": C["orange"],
            "Ours": C["green"],
        },
        "markers": {
            "AB-Effective": "o",
            "AB-Factor": "s",
            "Ours": "D",
        },
    }


# ============================================================
# Fig 1 — Tensor-TopK failure across tasks
# ============================================================
def fig1():
    fig = plt.figure(figsize=(8.8, 3.1))

    # ---- (a) GSM8K EM -------------------------------------------------
    ax = fig.add_subplot(1, 2, 1)
    methods = ["Tensor\nTop-K", "A/B Pair\nTop-K", "Dense\nFull"]
    em = [1.77, 28.00, 34.70]
    method_colors = [C["red"], C["blue"], C["gray"]]
    x = np.arange(len(methods))

    bars = ax.bar(x, em, color=method_colors, edgecolor="black", linewidth=0.4, width=0.5)
    for b, v in zip(bars, em):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Fixed positions
    ax.annotate("Catastrophic\nfailure\n(EM=1.77)",
                xy=(0, 4.1), xytext=(0, 10.8),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.0),
                fontsize=9, color=C["red"], fontweight="bold", ha="center")

    ax.annotate("A/B pair recovers\n+26.2 EM",
                xy=(1, 31), xytext=(1, 36),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.0),
                fontsize=8.5, color=C["blue"], fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("GSM8K Exact Match (%)")
    ax.set_ylim(0, 44)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.set_title("(a) Math Reasoning (GSM8K)", fontsize=10.5)
    ax.legend(
        handles=[
            Patch(facecolor=C["red"], edgecolor="black", label="Tensor Top-K"),
            Patch(facecolor=C["blue"], edgecolor="black", label="A/B Pair Top-K"),
            Patch(facecolor=C["gray"], edgecolor="black", label="Dense Full"),
        ],
        loc="upper left", fontsize=7.0, framealpha=0.92
    )

    # ---- (b) Dolly Generation (single Y, grouped bars) ---------------
    ax = fig.add_subplot(1, 2, 2)
    methods2 = ["Tensor\nTop-K", "A/B Pair\nTop-K"]
    genlen  = [127.79, 64.52]
    rougel  = [20.33, 35.26]
    if DOLLY_DENSE_GENLEN is not None and DOLLY_DENSE_ROUGEL is not None:
        methods2.append("Dense\nFull")
        genlen.append(DOLLY_DENSE_GENLEN)
        rougel.append(DOLLY_DENSE_ROUGEL)
    x2 = np.arange(len(methods2))
    w = 0.3

    method_colors2 = [C["red"], C["blue"]] + ([C["gray"]] if len(methods2) == 3 else [])
    b_gen = ax.bar(x2 - w/2, genlen, w, color=method_colors2,
                   edgecolor="black", linewidth=0.35)
    b_rouge = ax.bar(x2 + w/2, rougel, w, color="white",
                     edgecolor=method_colors2, linewidth=1.0, hatch="///")

    for b, v in zip(b_gen, genlen):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b, v in zip(b_rouge, rougel):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ax.axhline(y=128, color="red", linestyle=":", alpha=0.35, linewidth=0.8)
    # ax.text(0.97, 129.5, "generation limit", transform=ax.get_yaxis_transform(),
    #         ha="right", va="bottom", fontsize=7, color="red", alpha=0.55,
    #         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    ax.legend(
        handles=[
            Patch(facecolor="0.65", edgecolor="black", label="Generation Length"),
            Patch(facecolor="white", edgecolor="black", hatch="///", label="ROUGE-L"),
        ],
        loc="upper right", fontsize=7.5, framealpha=0.9
    )

    ax.set_xticks(x2)
    ax.set_xticklabels(methods2)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 155)
    ax.set_title("(b) Instruction Generation (Dolly-15K)", fontsize=10.5)

    fig.tight_layout()
    for fmt in ["pdf","png"]:
        fig.savefig(os.path.join(OUT, f"fig1_tensor_topk_failure.{fmt}"))
    plt.close(fig)
    print("[OK] Fig 1")


# ============================================================
# Fig 2 — Selection Concentration
# ============================================================
def fig2():
    rounds = [1, 2, 3, 4, 5]
    jac_eff  = [0.82, 1.00, 1.00, 1.00, 1.00]
    jac_fac  = [0.74, 0.84, 0.60, 0.55, 0.43]
    jac_ours = [0.19, 0.17, 0.18, 0.17, 0.16]
    uni_eff  = [24, 20, 20, 20, 20]
    uni_fac  = [30, 34, 32, 44, 52]
    uni_ours = [80, 80, 80, 84, 84]
    shared_eff  = [10, 20, 20, 20, 20]
    shared_fac  = [10, 12, 8, 6, 6]
    shared_ours = [0, 0, 0, 0, 0]

    fig = plt.figure(figsize=(9, 3.6))

    # ---- (a) Pairwise Jaccard ------------------------------------------
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(rounds, jac_eff,  "o-",  color=C["blue"],   lw=1.8, ms=6, label="AB-Effective")
    ax.plot(rounds, jac_fac,  "s--", color=C["orange"], lw=1.8, ms=6, label="AB-Factor")
    ax.plot(rounds, jac_ours, "D-",  color=C["green"],  lw=1.8, ms=6, label="Ours (SN-P1/P2)")

    ax.annotate("Jaccard = 1.0 (identical selection)\nfrom round 2 onwards",
                xy=(2.1, 0.98), xytext=(3.4, 0.68),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                fontsize=8, color=C["blue"], fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"))

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Pairwise Jaccard (Qwen2.5-14B)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(rounds)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_title("(a) Cross-Client Selection Overlap", fontsize=10.5)

    # ---- (b) Union + Fully Shared -------------------------------------
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(rounds, uni_eff,  "o-",  color=C["blue"],   lw=1.8, ms=6, label="AB-Effective (union)")
    ax.plot(rounds, uni_fac,  "s--", color=C["orange"], lw=1.8, ms=6, label="AB-Factor (union)")
    ax.plot(rounds, uni_ours, "D-",  color=C["green"],  lw=1.8, ms=6, label="Ours (union)")

    # Fully-shared as dashed lighter lines with markers
    ax.plot(rounds, shared_eff,  "o:",  color=C["blue"],   lw=1.2, ms=5, alpha=0.6,
            label="AB-Effective (fully shared)")
    ax.plot(rounds, shared_fac,  "s:",  color=C["orange"], lw=1.2, ms=5, alpha=0.6,
            label="AB-Factor (fully shared)")
    ax.plot(rounds, shared_ours, "D:",  color=C["green"],  lw=1.2, ms=5, alpha=0.6,
            label="Ours (fully shared)")

    ax.annotate("Ours covers 4× more\nunique modules", xy=(4.1, 83),
                xytext=(2.3, 60),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                fontsize=9, color=C["green"], fontweight="bold", ha="center")

    ax.annotate("AB-Eff: 20/20 modules\nshared by ALL clients",
                xy=(2, 20), xytext=(0.5, 38),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.8),
                fontsize=8, color=C["blue"], fontweight="bold", ha="center")

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("# Modules (Qwen2.5-14B)")
    ax.set_xticks(rounds)
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_title("(b) Module Coverage & Redundancy", fontsize=10.5)

    fig.tight_layout()
    for fmt in ["pdf","png"]:
        fig.savefig(os.path.join(OUT, f"fig2_selection_concentration.{fmt}"))
    plt.close(fig)
    print("[OK] Fig 2")


def fig2_v1_line_plus_summary():
    """Recommended paper version: one process view + one compact summary."""
    d = fig2_data()
    rounds = d["rounds"]
    methods = d["methods"]
    colors = d["colors"]
    markers = d["markers"]

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for name, ls in [("AB-Effective", "-"), ("AB-Factor", "--"), ("Ours", "-")]:
        ax.plot(
            rounds, d["jac"][name],
            marker=markers[name], linestyle=ls, color=colors[name],
            lw=1.8, ms=5.5, label=name if name != "Ours" else "Ours"
        )

    # ax.axhspan(0.85, 1.02, color=C["red"], alpha=0.06, zorder=0)
    ax.text(3.1, 0.90, "Jaccard=1:\nsame modules", color=C["red"],
            fontsize=7.5, ha="center", va="center")
    ax.text(3.1, 0.25, "low cross-client overlap", color=C["green"],
            fontsize=7.5, ha="center", va="center")

    ax.set_title("(a) Cross-client overlap", fontsize=10)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Pairwise Jaccard")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.48),
              fontsize=7.2, framealpha=0.92)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(methods))
    union_mean = np.array([d["union"][m].mean() for m in methods])
    shared_mean = np.array([d["shared"][m].mean() for m in methods])
    w = 0.34

    b1 = ax.bar(x - w/2, union_mean, width=w, color=[colors[m] for m in methods],
                alpha=0.82, edgecolor="black", linewidth=0.35, label="Covered modules")
    b2 = ax.bar(x + w/2, shared_mean, width=w, color="white",
                edgecolor=[colors[m] for m in methods], linewidth=1.2,
                hatch="///", label="Fully shared modules")

    for bars, vals in [(b1, union_mean), (b2, shared_mean)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 2.0, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.3, fontweight="bold")

    ax.annotate("4× more\ncoverage",
                xy=(2 - w/2, union_mean[2]), xytext=(1.2, 70),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                fontsize=7.5, color=C["green"], fontweight="bold", ha="center")
    ax.annotate("0\nfully\nshared\nmodules",
                xy=(2 + w/2, 0), xytext=(2.22, 17),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                fontsize=7.2, color=C["green"], fontweight="bold", ha="center")

    ax.set_title("(b) Coverage and redundancy", fontsize=10)
    ax.set_ylabel("# modules, mean over rounds")
    ax.set_xticks(x)
    ax.set_xticklabels(["AB-Effective", "AB-Factor", "Ours"], fontsize=6.8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig2_v1_line_plus_summary.{fmt}"))
    plt.close(fig)
    print("[OK] Fig 2 v1")


def fig2_v2_summary_only():
    """Compact summary version for a tight paper layout."""
    d = fig2_data()
    methods = d["methods"]
    colors = d["colors"]
    x = np.arange(len(methods))

    mean_jac = np.array([d["jac"][m].mean() for m in methods])
    mean_union = np.array([d["union"][m].mean() for m in methods])
    mean_shared = np.array([d["shared"][m].mean() for m in methods])

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.6))

    panels = [
        ("(a) Overlap", "Pairwise Jaccard ↓", mean_jac, (0, 1.08), "{:.2f}"),
        ("(b) Coverage", "# covered modules ↑", mean_union, (0, 105), "{:.1f}"),
        ("(c) Redundancy", "# fully shared modules ↓", mean_shared, (0, 24), "{:.1f}"),
    ]

    for ax, (title, ylabel, vals, ylim, fmt) in zip(axes, panels):
        bars = ax.bar(x, vals, width=0.55, color=[colors[m] for m in methods],
                      alpha=0.86, edgecolor="black", linewidth=0.35)
        for b, v in zip(bars, vals):
            offset = 0.03 * (ylim[1] - ylim[0])
            ax.text(b.get_x() + b.get_width()/2, v + offset, fmt.format(v),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(["AB-Eff.", "AB-Fac.", "Ours"], rotation=0)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.7)

    axes[0].annotate("magnitude-only", xy=(0, mean_jac[0]), xytext=(0.6, 0.55),
                     arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8),
                     fontsize=7.2, color=C["red"], ha="center")
    axes[1].annotate("4×", xy=(2, mean_union[2]), xytext=(1.35, 87),
                     arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                     fontsize=8.5, color=C["green"], fontweight="bold", ha="center")
    axes[2].annotate("0", xy=(2, mean_shared[2]), xytext=(2, 8),
                     arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                     fontsize=8.5, color=C["green"], fontweight="bold", ha="center")

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig2_v2_summary_only.{fmt}"))
    plt.close(fig)
    print("[OK] Fig 2 v2")


def fig2_v3_split_coverage_redundancy():
    """Keep time evolution, but split union and fully-shared to avoid six-line clutter."""
    d = fig2_data()
    rounds = d["rounds"]
    colors = d["colors"]
    markers = d["markers"]

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[:, 0])
    for name, ls in [("AB-Effective", "-"), ("AB-Factor", "--"), ("Ours", "-")]:
        ax.plot(rounds, d["jac"][name], marker=markers[name], linestyle=ls,
                color=colors[name], lw=1.8, ms=5.5, label=name)
    ax.set_title("(a) Cross-client overlap", fontsize=10)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Pairwise Jaccard")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    ax_u = fig.add_subplot(gs[0, 1])
    for name, ls in [("AB-Effective", "-"), ("AB-Factor", "--"), ("Ours", "-")]:
        ax_u.plot(rounds, d["union"][name], marker=markers[name], linestyle=ls,
                  color=colors[name], lw=1.8, ms=5.2)
    ax_u.set_title("(b) Covered modules", fontsize=10)
    ax_u.set_ylabel("# modules")
    ax_u.set_xticks(rounds)
    ax_u.set_ylim(0, 95)
    ax_u.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax_u.text(3.25, 77, "Ours ≈ 4× AB-Eff.", color=C["green"],
              fontsize=7.5, fontweight="bold")

    ax_s = fig.add_subplot(gs[1, 1])
    for name, ls in [("AB-Effective", "-"), ("AB-Factor", "--"), ("Ours", "-")]:
        ax_s.plot(rounds, d["shared"][name], marker=markers[name], linestyle=ls,
                  color=colors[name], lw=1.8, ms=5.2)
    ax_s.set_title("(c) Fully shared modules", fontsize=10)
    ax_s.set_xlabel("Federated round")
    ax_s.set_ylabel("# modules")
    ax_s.set_xticks(rounds)
    ax_s.set_ylim(-1, 23)
    ax_s.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax_s.text(2.8, 18, "AB-Eff. = all 20 modules\nfrom round 2", color=C["red"],
              fontsize=7.3, fontweight="bold", ha="center")

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig2_v3_split_coverage_redundancy.{fmt}"))
    plt.close(fig)
    print("[OK] Fig 2 v3")


# ============================================================
# Pillow fallback — used when matplotlib is unavailable
# ============================================================
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _pil_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_center(draw, xy, text, font, fill=(0, 0, 0), anchor="mm"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _paste_rotated_label(img, text, center_xy, font):
    label = Image.new("RGBA", (640, 64), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((320, 32), text, font=font, fill=(0, 0, 0), anchor="mm")
    rotated = label.rotate(90, expand=True)
    x = int(center_xy[0] - rotated.size[0] / 2)
    y = int(center_xy[1] - rotated.size[1] / 2)
    img.paste(rotated, (x, y), rotated)


def _draw_axes(draw, box, x_ticks, y_ticks, x_label, y_label, title, fonts):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(0, 0, 0), width=3)
    for yt in y_ticks:
        yy = y1 - (yt - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]) * (y1 - y0)
        draw.line([(x0, yy), (x1, yy)], fill=(228, 228, 228), width=2)
        draw.text((x0 - 12, yy), f"{yt:g}", font=fonts["tick"], fill=(0, 0, 0), anchor="rm")
    for xt in x_ticks:
        xx = x0 + (xt - x_ticks[0]) / (x_ticks[-1] - x_ticks[0]) * (x1 - x0)
        draw.line([(xx, y1), (xx, y1 + 9)], fill=(0, 0, 0), width=2)
        draw.text((xx, y1 + 18), f"{xt:g}", font=fonts["tick"], fill=(0, 0, 0), anchor="mt")
    draw.text(((x0 + x1) / 2, y1 + 58), x_label, font=fonts["label"], fill=(0, 0, 0), anchor="mm")
    label_img = Image.new("RGBA", (260, 44), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((130, 22), y_label, font=fonts["label"], fill=(0, 0, 0), anchor="mm")
    label_img = label_img.rotate(90, expand=True)
    return label_img


def _plot_line(draw, box, xs, ys, xlim, ylim, color, width=5, marker="circle", dash=False):
    x0, y0, x1, y1 = box
    pts = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        py = y1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        pts.append((px, py))
    if dash:
        for i in range(len(pts) - 1):
            x_a, y_a = pts[i]
            x_b, y_b = pts[i + 1]
            steps = 16
            for j in range(0, steps, 2):
                p0 = j / steps
                p1 = min((j + 1) / steps, 1)
                draw.line([(x_a + (x_b - x_a) * p0, y_a + (y_b - y_a) * p0),
                           (x_a + (x_b - x_a) * p1, y_a + (y_b - y_a) * p1)],
                          fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width, joint="curve")
    r = 12
    for px, py in pts:
        if marker == "square":
            draw.rectangle((px-r, py-r, px+r, py+r), fill=color, outline=(255, 255, 255), width=2)
        elif marker == "diamond":
            draw.polygon([(px, py-r-3), (px+r+3, py), (px, py+r+3), (px-r-3, py)],
                         fill=color, outline=(255, 255, 255))
        else:
            draw.ellipse((px-r, py-r, px+r, py+r), fill=color, outline=(255, 255, 255), width=2)


def _legend(draw, items, x, y, fonts):
    row_h = 31
    w = 250
    h = row_h * len(items) + 18
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(255, 255, 255), outline=(205, 205, 205), width=2)
    for idx, (label, color, marker) in enumerate(items):
        cy = y + 18 + idx * row_h
        _plot_line(draw, (x + 15, cy - 1, x + 65, cy + 1), [0, 1], [0, 0], (0, 1), (-1, 1), color, width=5, marker=marker)
        draw.text((x + 82, cy), label, font=fonts["tick"], fill=(0, 0, 0), anchor="lm")


def _bar_y(value, y0, y1, ymax):
    return y1 - value / ymax * (y1 - y0)


def pil_fig1():
    W, H = 2200, 820
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    fonts = {
        "panel": _pil_font(32, False),
        "label": _pil_font(28, False),
        "tick": _pil_font(24, False),
        "small": _pil_font(22, False),
        "bold": _pil_font(25, True),
    }
    red = _hex_to_rgb(C["red"]); blue = _hex_to_rgb(C["blue"]); gray = _hex_to_rgb(C["gray"])

    # (a) GSM8K
    left = (115, 125, 1040, 680)
    x0, y0, x1, y1 = left
    draw.text(((x0+x1)/2, 80), "(a) Math Reasoning (GSM8K)", font=fonts["panel"], fill=(0,0,0), anchor="mm")
    draw.rectangle(left, outline=(0,0,0), width=3)
    for yt in range(0, 41, 10):
        yy = _bar_y(yt, y0, y1, 45)
        draw.line([(x0, yy), (x1, yy)], fill=(230,230,230), width=2)
        draw.text((x0-12, yy), str(yt), font=fonts["tick"], fill=(0,0,0), anchor="rm")
    methods = ["Tensor\nTop-K", "A/B Pair\nTop-K", "Dense\nFull"]
    vals = [1.77, 28.0, 34.7]
    cols = [red, blue, gray]
    xs = np.linspace(x0+130, x1-130, 3)
    bw = 115
    for xi, v, col, lab in zip(xs, vals, cols, methods):
        yy = _bar_y(v, y0, y1, 45)
        draw.rectangle((xi-bw/2, yy, xi+bw/2, y1), fill=col, outline=(0,0,0), width=2)
        draw.text((xi, yy-18), f"{v:.1f}", font=fonts["bold"], fill=(0,0,0), anchor="mm")
        draw.multiline_text((xi, y1+28), lab, font=fonts["tick"], fill=(0,0,0), anchor="ma", align="center", spacing=2)
    _paste_rotated_label(img, "GSM8K Exact Match (%)", (42, (y0+y1)/2), fonts["label"])
    draw.text((xs[0], 485), "Catastrophic\nfailure", font=fonts["bold"], fill=red, anchor="mm", align="center")
    draw.line([(xs[0], 520), (xs[0], _bar_y(vals[0], y0, y1, 45)+8)], fill=red, width=4)
    draw.text((xs[1], 205), "A/B pair recovers\n+26.2 EM", font=fonts["bold"], fill=blue, anchor="mm", align="center")
    draw.line([(xs[1], 240), (xs[1], _bar_y(vals[1], y0, y1, 45)-5)], fill=blue, width=3)
    lx, ly = x0 + 30, y0 + 28
    for j, (name, col) in enumerate([("Tensor Top-K", red), ("A/B Pair Top-K", blue), ("Dense Full", gray)]):
        yy = ly + j * 35
        draw.rectangle((lx, yy, lx + 28, yy + 20), fill=col, outline=(0,0,0))
        draw.text((lx + 42, yy + 10), name, font=fonts["small"], fill=(0,0,0), anchor="lm")

    # (b) Dolly
    right = (1250, 125, 2145, 680)
    x0, y0, x1, y1 = right
    draw.text(((x0+x1)/2, 80), "(b) Instruction Generation (Dolly-15K)", font=fonts["panel"], fill=(0,0,0), anchor="mm")
    draw.rectangle(right, outline=(0,0,0), width=3)
    for yt in range(0, 141, 20):
        yy = _bar_y(yt, y0, y1, 150)
        draw.line([(x0, yy), (x1, yy)], fill=(230,230,230), width=2)
        draw.text((x0-12, yy), str(yt), font=fonts["tick"], fill=(0,0,0), anchor="rm")
    methods2 = ["Tensor\nTop-K", "A/B Pair\nTop-K"]
    gen = [127.79, 64.52]
    rouge = [20.33, 35.26]
    if DOLLY_DENSE_GENLEN is not None and DOLLY_DENSE_ROUGEL is not None:
        methods2.append("Dense\nFull")
        gen.append(DOLLY_DENSE_GENLEN)
        rouge.append(DOLLY_DENSE_ROUGEL)
    xs2 = np.linspace(x0+165, x1-165, len(methods2))
    bw = 78
    for i, xi in enumerate(xs2):
        gy = _bar_y(gen[i], y0, y1, 150)
        ry = _bar_y(rouge[i], y0, y1, 150)
        gcol = [red, blue, gray][min(i, 2)]
        rcol = gcol
        draw.rectangle((xi-bw, gy, xi-8, y1), fill=gcol, outline=(0,0,0), width=2)
        draw.rectangle((xi+8, ry, xi+bw, y1), fill=(255,255,255), outline=rcol, width=4)
        for yy in np.arange(ry + 6, y1, 13):
            draw.line([(xi+8, yy), (xi+bw, yy-18)], fill=rcol, width=2)
        draw.text((xi-bw/2-4, gy-18), f"{gen[i]:.0f}", font=fonts["bold"], fill=(0,0,0), anchor="mm")
        draw.text((xi+bw/2+4, ry-18), f"{rouge[i]:.1f}", font=fonts["bold"], fill=(0,0,0), anchor="mm")
        draw.multiline_text((xi, y1+28), methods2[i], font=fonts["tick"], fill=(0,0,0), anchor="ma", align="center", spacing=2)
    yy128 = _bar_y(128, y0, y1, 150)
    draw.line([(x0, yy128), (x1, yy128)], fill=(255,120,120), width=2)
    draw.text((x1-145, yy128-22), "generation limit", font=fonts["small"], fill=(220,80,80), anchor="mm")
    _paste_rotated_label(img, "Score", (1192, (y0+y1)/2), fonts["label"])
    lgx, lgy = x1 - 385, y0 + 95
    draw.rectangle((lgx, lgy, lgx+35, lgy+25), fill=gray, outline=(0,0,0))
    draw.text((lgx+52, lgy-1), "Generation Length", font=fonts["tick"], fill=(0,0,0))
    draw.rectangle((lgx, lgy+42, lgx+35, lgy+67), fill=(255,255,255), outline=(0,0,0), width=3)
    for yy in range(lgy+47, lgy+67, 8):
        draw.line([(lgx, yy), (lgx+35, yy-16)], fill=(0,0,0), width=2)
    draw.text((lgx+52, lgy+41), "ROUGE-L", font=fonts["tick"], fill=(0,0,0))

    out = os.path.join(OUT, "fig1_tensor_topk_failure.png")
    img.save(out)
    print("[OK] Fig 1 PIL")


def pil_fig2_v1_line_plus_summary():
    d = fig2_data()
    W, H = 2200, 900
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    fonts = {
        "title": _pil_font(34, True),
        "panel": _pil_font(30, False),
        "label": _pil_font(27, False),
        "tick": _pil_font(23, False),
        "small": _pil_font(22, False),
        "small_bold": _pil_font(22, True),
    }
    red = _hex_to_rgb(C["red"]); orange = _hex_to_rgb(C["orange"]); green = _hex_to_rgb(C["green"])

    left = (125, 170, 1030, 720)
    right = (1260, 170, 2100, 720)
    _draw_center(draw, ((left[0]+left[2])/2, 125), "(a) Cross-client overlap", fonts["panel"])
    _draw_center(draw, ((right[0]+right[2])/2, 125), "(b) Coverage and redundancy", fonts["panel"])

    x0, y0, x1, y1 = left
    draw.rectangle(left, outline=(0, 0, 0), width=3)
    for yt in np.arange(0, 1.01, 0.2):
        yy = y1 - yt / 1.05 * (y1-y0)
        draw.line([(x0, yy), (x1, yy)], fill=(230, 230, 230), width=2)
        draw.text((x0-12, yy), f"{yt:.1f}", font=fonts["tick"], fill=(0,0,0), anchor="rm")
    for xt in d["rounds"]:
        xx = x0 + (xt-1)/4*(x1-x0)
        draw.text((xx, y1+32), str(int(xt)), font=fonts["tick"], fill=(0,0,0), anchor="mm")
    draw.text(((x0+x1)/2, y1+82), "Federated round", font=fonts["label"], fill=(0,0,0), anchor="mm")
    ylab = Image.new("RGBA", (390, 52), (255,255,255,0))
    yd = ImageDraw.Draw(ylab)
    yd.text((195, 26), "Pairwise Jaccard", font=fonts["label"], fill=(0,0,0), anchor="mm")
    img.paste(ylab.rotate(90, expand=True), (18, 310), ylab.rotate(90, expand=True))

    _plot_line(draw, left, d["rounds"], d["jac"]["AB-Effective"], (1, 5), (0, 1.05), red, marker="circle")
    _plot_line(draw, left, d["rounds"], d["jac"]["AB-Factor"], (1, 5), (0, 1.05), orange, marker="square", dash=True)
    _plot_line(draw, left, d["rounds"], d["jac"]["Ours"], (1, 5), (0, 1.05), green, marker="diamond")
    _legend(draw, [("AB-Effective", red, "circle"), ("AB-Factor", orange, "square"), ("Ours", green, "diamond")], 740, 205, fonts)
    draw.text((650, 355), "Jaccard=1:\nsame modules", font=fonts["small_bold"], fill=red, anchor="mm")
    draw.line([(595, 330), (420, 195)], fill=(0,0,0), width=3)
    draw.polygon([(420,195), (434,198), (425,209)], fill=(0,0,0))
    draw.text((785, 535), "complementary\nselection", font=fonts["small_bold"], fill=green, anchor="mm")

    x0, y0, x1, y1 = right
    draw.rectangle(right, outline=(0, 0, 0), width=3)
    for yt in range(0, 101, 20):
        yy = y1 - yt/100*(y1-y0)
        draw.line([(x0, yy), (x1, yy)], fill=(230,230,230), width=2)
        draw.text((x0-12, yy), str(yt), font=fonts["tick"], fill=(0,0,0), anchor="rm")
    methods = d["methods"]
    union_mean = [d["union"][m].mean() for m in methods]
    shared_mean = [d["shared"][m].mean() for m in methods]
    xs = [x0 + 150, x0 + 420, x0 + 690]
    bw = 58
    for i, m in enumerate(methods):
        col = {"AB-Effective": red, "AB-Factor": orange, "Ours": green}[m]
        ux = xs[i] - 35
        sx = xs[i] + 35
        uh = union_mean[i] / 100 * (y1-y0)
        sh = shared_mean[i] / 100 * (y1-y0)
        draw.rectangle((ux-bw/2, y1-uh, ux+bw/2, y1), fill=col, outline=(0,0,0), width=2)
        draw.rectangle((sx-bw/2, y1-sh, sx+bw/2, y1), fill=(255,255,255), outline=col, width=4)
        for yy in np.arange(y1-sh+6, y1, 13):
            draw.line([(sx-bw/2, yy), (sx+bw/2, yy-18)], fill=col, width=2)
        draw.text((ux, y1-uh-18), f"{union_mean[i]:.1f}", font=fonts["small_bold"], fill=(0,0,0), anchor="mm")
        draw.text((sx, y1-sh-18 if sh > 15 else y1-sh-26), f"{shared_mean[i]:.1f}", font=fonts["small_bold"], fill=(0,0,0), anchor="mm")
        draw.text((xs[i], y1+38), ["AB-Effective", "AB-Factor", "Ours"][i], font=fonts["tick"], fill=(0,0,0), anchor="mm")
    draw.text(((x0+x1)/2, y1+82), "Method", font=fonts["label"], fill=(0,0,0), anchor="mm")
    _paste_rotated_label(img, "# modules, mean over rounds", (1188, (y0+y1)/2), fonts["label"])
    draw.text((1780, 235), "Covered modules", font=fonts["tick"], fill=(0,0,0), anchor="lm")
    draw.rectangle((1725, 220, 1765, 245), fill=(160,160,160), outline=(0,0,0))
    draw.text((1780, 280), "Fully shared modules", font=fonts["tick"], fill=(0,0,0), anchor="lm")
    draw.rectangle((1725, 265, 1765, 290), fill=(255,255,255), outline=(0,0,0), width=3)
    draw.text((1740, 355), "4× more\ncoverage", font=fonts["small_bold"], fill=green, anchor="mm")
    draw.line([(1680, 380), (1958, 270)], fill=green, width=3)
    draw.text((1795, 575), "0 fully-shared", font=fonts["small_bold"], fill=green, anchor="mm")

    out = os.path.join(OUT, "fig2_v1_line_plus_summary.png")
    img.save(out)
    print("[OK] Fig 2 v1 PIL")


def pil_fig2_v2_summary_only():
    d = fig2_data()
    W, H = 2300, 760
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    fonts = {"title": _pil_font(34, True), "panel": _pil_font(28), "label": _pil_font(25), "tick": _pil_font(22), "bold": _pil_font(23, True)}
    red = _hex_to_rgb(C["red"]); orange = _hex_to_rgb(C["orange"]); green = _hex_to_rgb(C["green"])
    colors = [red, orange, green]
    methods = d["methods"]
    vals = [
        [d["jac"][m].mean() for m in methods],
        [d["union"][m].mean() for m in methods],
        [d["shared"][m].mean() for m in methods],
    ]
    titles = ["(a) Overlap", "(b) Coverage", "(c) Redundancy"]
    labels = ["Pairwise Jaccard ↓", "# covered modules ↑", "# fully shared modules ↓"]
    ylims = [1.05, 100, 24]
    fmts = ["{:.2f}", "{:.1f}", "{:.1f}"]
    boxes = [(120, 155, 690, 590), (875, 155, 1445, 590), (1630, 155, 2200, 590)]
    for pi, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.text(((x0+x1)/2, 112), titles[pi], font=fonts["panel"], anchor="mm")
        draw.rectangle(box, outline=(0,0,0), width=3)
        for t in np.linspace(0, ylims[pi], 5):
            yy = y1 - t/ylims[pi]*(y1-y0)
            draw.line([(x0, yy), (x1, yy)], fill=(230,230,230), width=2)
            label = f"{t:.1f}" if ylims[pi] <= 2 else f"{int(t)}"
            draw.text((x0-12, yy), label, font=fonts["tick"], anchor="rm")
        xs = np.linspace(x0+110, x1-110, 3)
        for i, v in enumerate(vals[pi]):
            bh = v/ylims[pi]*(y1-y0)
            draw.rectangle((xs[i]-45, y1-bh, xs[i]+45, y1), fill=colors[i], outline=(0,0,0), width=2)
            draw.text((xs[i], y1-bh-18), fmts[pi].format(v), font=fonts["bold"], anchor="mm")
            draw.text((xs[i], y1+24), ["AB-Eff.", "AB-Fac.", "Ours"][i], font=fonts["tick"], anchor="mt")
        draw.text(((x0+x1)/2, y1+68), labels[pi], font=fonts["label"], anchor="mm")
    out = os.path.join(OUT, "fig2_v2_summary_only.png")
    img.save(out)
    print("[OK] Fig 2 v2 PIL")


def pil_fig2_v3_split_coverage_redundancy():
    d = fig2_data()
    W, H = 2150, 1260
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    fonts = {"title": _pil_font(34, True), "panel": _pil_font(29), "label": _pil_font(25), "tick": _pil_font(22), "bold": _pil_font(22, True)}
    red = _hex_to_rgb(C["red"]); orange = _hex_to_rgb(C["orange"]); green = _hex_to_rgb(C["green"])
    boxes = {
        "jac": (120, 165, 1040, 1110),
        "union": (1240, 165, 2050, 565),
        "shared": (1240, 710, 2050, 1110),
    }
    meta = [
        ("jac", "(a) Cross-client overlap", "Pairwise Jaccard", (0, 1.05), "jac"),
        ("union", "(b) Covered modules", "# modules", (0, 90), "union"),
        ("shared", "(c) Fully shared modules", "# modules", (0, 22), "shared"),
    ]
    for key, title, ylabel, ylim, field in meta:
        box = boxes[key]
        x0, y0, x1, y1 = box
        draw.text(((x0+x1)/2, y0-45), title, font=fonts["panel"], anchor="mm")
        draw.rectangle(box, outline=(0,0,0), width=3)
        for t in np.linspace(ylim[0], ylim[1], 6):
            yy = y1 - (t-ylim[0])/(ylim[1]-ylim[0])*(y1-y0)
            draw.line([(x0, yy), (x1, yy)], fill=(230,230,230), width=2)
            label = f"{t:.1f}" if ylim[1] <= 2 else f"{int(t)}"
            draw.text((x0-12, yy), label, font=fonts["tick"], anchor="rm")
        for xt in d["rounds"]:
            xx = x0 + (xt-1)/4*(x1-x0)
            draw.text((xx, y1+20), str(int(xt)), font=fonts["tick"], anchor="mt")
        draw.text(((x0+x1)/2, y1+68), "Federated round", font=fonts["label"], anchor="mm")
        draw.text((x0-78, (y0+y1)/2), ylabel, font=fonts["label"], anchor="mm")
        _plot_line(draw, box, d["rounds"], d[field]["AB-Effective"], (1,5), ylim, red, marker="circle")
        _plot_line(draw, box, d["rounds"], d[field]["AB-Factor"], (1,5), ylim, orange, marker="square", dash=True)
        _plot_line(draw, box, d["rounds"], d[field]["Ours"], (1,5), ylim, green, marker="diamond")
    _legend(draw, [("AB-Effective", red, "circle"), ("AB-Factor", orange, "square"), ("Ours", green, "diamond")], 180, 220, fonts)
    draw.text((1720, 430), "Ours ≈ 4× AB-Eff.", font=fonts["bold"], fill=green, anchor="mm")
    draw.text((1660, 870), "AB-Eff. fully shares\nall 20 modules", font=fonts["bold"], fill=red, anchor="mm")
    out = os.path.join(OUT, "fig2_v3_split_coverage_redundancy.png")
    img.save(out)
    print("[OK] Fig 2 v3 PIL")


if __name__ == "__main__":
    print("Generating motivation figures...")
    if HAS_MPL:
        fig1()
        fig2()
        fig2_v1_line_plus_summary()
        fig2_v2_summary_only()
        fig2_v3_split_coverage_redundancy()
    else:
        print("matplotlib is unavailable; using Pillow fallback for Fig. 2 PNG drafts.")
        pil_fig1()
        pil_fig2_v1_line_plus_summary()
        pil_fig2_v2_summary_only()
        pil_fig2_v3_split_coverage_redundancy()
    print(f"Done → {OUT}")
