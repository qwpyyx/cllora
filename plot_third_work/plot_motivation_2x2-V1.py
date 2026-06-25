#!/usr/bin/env python3
"""
INFOCOM 2027 Motivation Figure: aligned 2x2 composite layout.

Panels:
  (a) GSM8K Tensor Top-K failure
  (b) Dolly-15K generation behavior
  (c) Cross-client selection overlap
  (d) Coverage and redundancy

This script intentionally produces one combined figure so the four panels
share the same grid geometry when inserted into the paper.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


matplotlib.rcParams['pdf.fonttype'] = 42

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_motivation")
os.makedirs(OUT, exist_ok=True)

C = {
    "red": "#d62728",
    "orange": "#ff7f0e",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "gray": "#7f7f7f",
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12.0,
        "axes.titlesize": 10.8,
        "axes.labelsize": 10.0,
        "legend.fontsize": 8.3,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })


def fig2_data():
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


def panel_gsm8k(ax):
    methods = ["Tensor\nTop-K", "A/B Pair\nTop-K", "Dense\nFull"]
    em = np.array([1.77, 28.00, 34.70])
    colors = [C["red"], C["blue"], C["gray"]]
    x = np.arange(len(methods))

    bars = ax.bar(x, em, width=0.55, color=colors, edgecolor="black", linewidth=0.45)
    for b, v in zip(bars, em):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.annotate("Catastrophic\nfailure",
                xy=(0, 5), xytext=(0, 11.5),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.9),
                fontsize=8.5, color=C["red"], fontweight="bold", ha="center")
    ax.annotate("A/B pair\nrecovers\n+26.2 EM",
                xy=(1, 31.0), xytext=(1, 37.0),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.9),
                fontsize=8.5, color=C["blue"], fontweight="bold", ha="center")

    ax.set_title("(a) Math reasoning (GSM8K)")
    ax.set_ylabel("Exact match (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 44)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(
        handles=[
            Patch(facecolor=C["red"], edgecolor="black", label="Tensor Top-K"),
            Patch(facecolor=C["blue"], edgecolor="black", label="A/B Pair Top-K"),
            Patch(facecolor=C["gray"], edgecolor="black", label="Dense Full"),
        ],
        loc="upper left", framealpha=0.92, borderpad=0.35,
        handlelength=1.4, labelspacing=0.25,
    )


def panel_dolly(ax):
    methods = ["Tensor\nTop-K", "A/B Pair\nTop-K", "Dense\nFull"]
    genlen = np.array([127.79, 64.52, 51.6567])
    rougel = np.array([20.3345, 35.2605, 35.1083])
    colors = [C["red"], C["blue"], C["gray"]]
    x = np.arange(len(methods))
    w = 0.30

    b_gen = ax.bar(x - w / 2, genlen, width=w, color=colors,
                   edgecolor="black", linewidth=0.45)
    b_rouge = ax.bar(x + w / 2, rougel, width=w, color="white",
                     edgecolor=colors, linewidth=1.0, hatch="///")

    for b, v in zip(b_gen, genlen):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for b, v in zip(b_rouge, rougel):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_title("(b) Instruction generation (Dolly-15K)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 155)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(
        handles=[
            Patch(facecolor="0.65", edgecolor="black", label="Generation length"),
            Patch(facecolor="white", edgecolor="black", hatch="///", label="ROUGE-L"),
        ],
        loc="upper right", framealpha=0.92, borderpad=0.35,
        handlelength=1.4, labelspacing=0.25,
    )


def panel_overlap(ax):
    d = fig2_data()
    rounds = d["rounds"]
    for name, ls in [("AB-Effective", "-"), ("AB-Factor", "--"), ("Ours", "-")]:
        ax.plot(
            rounds, d["jac"][name],
            marker=d["markers"][name],
            linestyle=ls,
            color=d["colors"][name],
            lw=1.9,
            ms=5.0,
            label=name,
        )

    ax.text(3.05, 0.91, "Jaccard=1:\nsame modules",
            color=C["red"], fontsize=10.5, ha="center", fontweight="bold", va="center")
    ax.text(3.00, 0.25, "low cross-client overlap",
            color=C["green"], fontsize=10.5, ha="center", fontweight="bold", va="center")

    ax.set_title("(c) Cross-client overlap")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Pairwise Jaccard")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(loc="center left", framealpha=0.92, borderpad=0.35,
              handlelength=1.5, labelspacing=0.25)


def panel_coverage(ax):
    d = fig2_data()
    methods = d["methods"]
    colors = d["colors"]
    x = np.arange(len(methods))
    w = 0.34
    union_mean = np.array([d["union"][m].mean() for m in methods])
    shared_mean = np.array([d["shared"][m].mean() for m in methods])

    b1 = ax.bar(x - w / 2, union_mean, width=w,
                color=[colors[m] for m in methods], alpha=0.82,
                edgecolor="black", linewidth=0.45, label="Covered modules")
    b2 = ax.bar(x + w / 2, shared_mean, width=w, color="white",
                edgecolor=[colors[m] for m in methods], linewidth=1.1,
                hatch="///", label="Fully shared modules")

    for bars, vals in [(b1, union_mean), (b2, shared_mean)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    ax.annotate("4x more\ncoverage",
                xy=(2 - w / 2, union_mean[2]), xytext=(1.18, 70),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                fontsize=10.5, color=C["green"], fontweight="bold", ha="center")
    ax.annotate("0 fully\nshared",
                xy=(2 + w / 2, 0), xytext=(2.25, 20),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8),
                fontsize=10.5, color=C["green"], fontweight="bold", ha="center")

    ax.set_title("(d) Coverage and redundancy")
    ax.set_ylabel("# modules, mean over rounds")
    ax.set_xticks(x)
    ax.set_xticklabels(["AB-Effective", "AB-Factor", "FedSP-LoRA"])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", framealpha=0.92, borderpad=0.35,
              handlelength=1.5, labelspacing=0.25)


def make_figure():
    set_style()

    # 7.16 in fits a full-width two-column IEEE figure.
    # Height leaves enough room for readable labels after the paper scales the figure.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.16, 5.75),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.025, wspace=0.08, hspace=0.08)

    panel_gsm8k(axes[0, 0])
    panel_dolly(axes[0, 1])
    panel_overlap(axes[1, 0])
    panel_coverage(axes[1, 1])

    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3.0)

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"motivation_2x2.{fmt}"))
    plt.close(fig)
    print(f"[OK] saved to {OUT}/motivation_2x2.[pdf|png]")


if __name__ == "__main__":
    make_figure()
