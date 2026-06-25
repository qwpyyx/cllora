#!/usr/bin/env python3
"""
INFOCOM 2027 — Motivation Figures (2-Jump, clean 2+2 layout)
Fig 1: (a) GSM8K EM, (b) Dolly Gen Length + ROUGE-L
Fig 2: (a) Jaccard per round, (b) Union + Fully Shared modules
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

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


# ============================================================
# Fig 1 — Tensor-TopK failure across tasks
# ============================================================
def fig1():
    fig = plt.figure(figsize=(9, 3.5))

    # ---- (a) GSM8K EM -------------------------------------------------
    ax = fig.add_subplot(1, 2, 1)
    methods = ["Tensor\nTop-K", "A/B Pair\nTop-K", "Q/V Block\nTop-K", "Dense\nFull"]
    em = [1.77, 28.00, 28.08, 34.70]
    colors = [C["red"], C["purple"], C["green"], C["gray"]]
    x = np.arange(4)

    bars = ax.bar(x, em, color=colors, edgecolor="black", linewidth=0.4, width=0.5)
    for b, v in zip(bars, em):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Fixed positions
    ax.annotate("Catastrophic\nfailure\n(EM=1.77)",
                xy=(0, 2), xytext=(0, 14),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.0),
                fontsize=9, color=C["red"], fontweight="bold", ha="center")

    ax.annotate("A/B pair recovers\n+26.2 EM (80.7% of Dense)",
                xy=(1, 28), xytext=(1, 35),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=1.0),
                fontsize=8.5, color=C["purple"], fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("GSM8K Exact Match (%)")
    ax.set_ylim(0, 44)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.set_title("(a) Math Reasoning (GSM8K)", fontsize=10.5)

    # ---- (b) Dolly Generation (single Y, grouped bars) ---------------
    ax = fig.add_subplot(1, 2, 2)
    methods2 = ["Tensor\nTop-K", "A/B Pair\nTop-K"]
    genlen  = [127.79, 64.52]
    rougel  = [20.33, 35.26]
    x2 = np.arange(2)
    w = 0.3

    b_gen = ax.bar(x2 - w/2, genlen, w, color=[C["red"], C["purple"]],
                   edgecolor="black", linewidth=0.3, label="Generation Length")
    b_rouge = ax.bar(x2 + w/2, rougel, w, color=[C["orange"], C["blue"]],
                     edgecolor="black", linewidth=0.3, label="ROUGE-L")

    for b, v in zip(b_gen, genlen):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b, v in zip(b_rouge, rougel):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=128, color="red", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.text(1.1, 125, "generation limit", fontsize=7, color="red", alpha=0.5)

    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    ax.set_xticks(x2)
    ax.set_xticklabels(methods2)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 155)
    ax.set_title("(b) Instruction Generation (Dolly-15K)", fontsize=10.5)

    fig.suptitle("Fig. 1: Tensor-Level Top-K Catastrophically Fails Across Tasks\n"
                 "(Qwen2.5-14B, 12.5% budget, 3 seeds: 28/42/45)",
                 fontsize=10, y=1.04, fontweight="bold")
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
            label="AB-Eff. (fully shared)")
    ax.plot(rounds, shared_fac,  "s:",  color=C["orange"], lw=1.2, ms=5, alpha=0.6,
            label="AB-Fac. (fully shared)")
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

    fig.suptitle("Fig. 2: Magnitude-Only Selection Causes Extreme Cross-Client Concentration\n"
                 "(Qwen2.5-14B, GSM8K, 12.5% budget, seed 42. Same pattern on Llama-3.1-8B: AB-Eff. Jaccard=0.60, Ours=0.18)",
                 fontsize=9.5, y=1.04, fontweight="bold")
    fig.tight_layout()
    for fmt in ["pdf","png"]:
        fig.savefig(os.path.join(OUT, f"fig2_selection_concentration.{fmt}"))
    plt.close(fig)
    print("[OK] Fig 2")


if __name__ == "__main__":
    print("Generating motivation figures...")
    fig1()
    fig2()
    print(f"Done → {OUT}")
