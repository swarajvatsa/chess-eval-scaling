#!/usr/bin/env python3
"""
make_plots.py — Regenerate every figure in the analysis from the CSVs in data/.

All figures are dark-themed, high-resolution PNGs written to ../docs/images/.
Run:  python analysis/make_plots.py
Deps: matplotlib, numpy  (pip install matplotlib numpy)

The plots tell the study's story in five beats:
  1. loss_curves.png        bigger nets reach lower loss (a better raw judge)
  2. elo_vs_timecontrol.png strength vs thinking time, with confidence bands
  3. small_net_vs_time.png  the small net climbs as it gets more time
  4. catchup_slopes.png     the big net's curve is steeper — it closes the gap
  5. cores_scaling.png      more cores -> more search (nodes), strength flat in noise
"""

import csv
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)

# ---- dark, high-definition house style -------------------------------------
BG = "#0d1117"        # near-black canvas (GitHub dark)
PANEL = "#161b22"     # slightly lighter axes face
FG = "#e6edf3"        # light text
GRID = "#30363d"      # subtle gridlines
# an ordered palette: cool blues for compact nets, warm orange for the deep net
PALETTE = {
    "compact_tiny":     "#7ee787",  # green  (the surprising winner)
    "compact_small":    "#58a6ff",  # blue
    "compact_champion": "#79c0ff",  # light blue
    "compact_wide":     "#a5d6ff",  # pale blue
    "deep_b4":          "#ff7b72",  # red/orange (the big slow judge)
}
LABEL = {
    "compact_tiny": "compact · 102K",
    "compact_small": "compact · 427K",
    "compact_champion": "compact · 1.05M",
    "compact_wide": "compact · 2.10M",
    "deep_b4": "deep · 2.50M",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 17, "axes.titleweight": "bold",
    "figure.dpi": 200, "savefig.dpi": 200,
})


def _style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


# ---- 1. loss curves --------------------------------------------------------
def loss_curves():
    rows = read_csv("loss_curves.csv")
    epochs = [int(r["epoch"]) for r in rows]
    nets = ["compact_tiny", "compact_small", "compact_champion", "compact_wide", "deep_b4"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for net in nets:
        y = [float(r[net]) for r in rows]
        ax.plot(epochs, y, color=PALETTE[net], lw=2.6, label=LABEL[net])
        ax.scatter([epochs[-1]], [y[-1]], color=PALETTE[net], s=28, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("validation loss  (lower = better judge)")
    ax.set_title("Bigger networks learn to judge positions better")
    ax.legend(frameon=False, loc="upper right")
    ax.annotate("the biggest net's first epoch already\nbeats the smallest net's final epoch",
                xy=(1, 0.0287), xytext=(6, 0.045), color=FG, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=FG, alpha=0.7))
    _style(ax)
    _save(fig, "loss_curves.png")


# ---- 2. Elo vs time control, with 90% CI bands -----------------------------
def elo_vs_timecontrol():
    rows = read_csv("elo_vs_timecontrol.csv")
    nets = ["compact_tiny", "compact_small", "compact_champion", "compact_wide", "deep_b4"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for net in nets:
        rs = [r for r in rows if r["net"] == net]
        x = [float(r["movetime_s"]) for r in rs]
        y = [float(r["elo"]) for r in rs]
        lo = [float(r["elo_lo"]) for r in rs]
        hi = [float(r["elo_hi"]) for r in rs]
        ax.fill_between(x, lo, hi, color=PALETTE[net], alpha=0.13, linewidth=0)
        ax.plot(x, y, color=PALETTE[net], lw=2.6, marker="o", ms=5, label=LABEL[net])
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("thinking time per move  (seconds, log scale)")
    ax.set_ylabel("absolute Elo  (shaded = 90% confidence)")
    ax.set_title("Strength vs thinking time — the deep net starts last, climbs fastest")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    _save(fig, "elo_vs_timecontrol.png")


# ---- 3. the small net rises with time --------------------------------------
def small_net_vs_time():
    rows = [r for r in read_csv("elo_vs_timecontrol.csv") if r["net"] == "compact_tiny"]
    x = [float(r["movetime_s"]) for r in rows]
    y = [float(r["elo"]) for r in rows]
    lo = [float(r["elo_lo"]) for r in rows]
    hi = [float(r["elo_hi"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(x, lo, hi, color=PALETTE["compact_tiny"], alpha=0.18, linewidth=0)
    ax.plot(x, y, color=PALETTE["compact_tiny"], lw=3.0, marker="o", ms=7,
            label="compact · 102K (the smallest net)")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=FG, fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("thinking time per move  (seconds, log scale)")
    ax.set_ylabel("absolute Elo  (shaded = 90% confidence)")
    ax.set_title("Give the small net more time and it gets much stronger")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    _save(fig, "small_net_vs_time.png")


# ---- 4. catch-up slopes ----------------------------------------------------
def catchup_slopes():
    srows = read_csv("catchup_slopes.csv")
    erows = read_csv("elo_vs_timecontrol.csv")
    nets = ["compact_tiny", "compact_small", "compact_champion", "compact_wide", "deep_b4"]
    slope = {r["net"]: float(r["slope_elo_per_decade"]) for r in srows}
    fig, ax = plt.subplots(figsize=(12, 7))
    for net in nets:
        rs = [r for r in erows if r["net"] == net]
        x = np.array([float(r["movetime_s"]) for r in rs])
        y = np.array([float(r["elo"]) for r in rs])
        ax.scatter(x, y, color=PALETTE[net], s=34, zorder=5)
        # least-squares fit Elo = a + b*log10(t)
        lx = np.log10(x)
        b, a = np.polyfit(lx, y, 1)
        xf = np.array([0.3, 30.0])
        ax.plot(xf, a + b * np.log10(xf), color=PALETTE[net], lw=2.4,
                label=f"{LABEL[net]}   +{slope[net]:.0f} Elo / 10x time")
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("thinking time per move  (seconds, log scale)")
    ax.set_ylabel("absolute Elo")
    ax.set_title("The deep net's curve is steepest — it gains the most per extra second")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    _style(ax)
    _save(fig, "catchup_slopes.png")


# ---- 5. cores scaling: nodes rise, Elo flat --------------------------------
def cores_scaling():
    rows = read_csv("cores_scaling.csv")
    cores = [int(r["cores"]) for r in rows]
    nodes = [float(r["avg_nodes_per_move"]) / 1e6 for r in rows]
    elo = [float(r["elo"]) for r in rows]
    lo = [float(r["elo_lo"]) for r in rows]
    hi = [float(r["elo_hi"]) for r in rows]
    x = np.arange(len(cores))

    fig, (axN, axE) = plt.subplots(1, 2, figsize=(15, 7))
    # left: nodes per move (bars, log-y) — the clean mechanism
    axN.bar(x, nodes, color="#58a6ff", width=0.6)
    axN.set_yscale("log")
    axN.set_xticks(x); axN.set_xticklabels([f"{c}" for c in cores])
    axN.set_xlabel("cores per move")
    axN.set_ylabel("nodes searched per move  (millions, log scale)")
    axN.set_title("More cores → more search (clean, ~linear)")
    for xi, n in zip(x, nodes):
        axN.annotate(f"{n:.0f}M", (xi, n), textcoords="offset points",
                     xytext=(0, 6), ha="center", color=FG, fontsize=11)
    _style(axN)
    # right: Elo with CI — flat within noise
    axE.errorbar(x, elo, yerr=[np.array(elo) - np.array(lo), np.array(hi) - np.array(elo)],
                 fmt="o", color="#7ee787", ecolor="#7ee787", elinewidth=2, capsize=6, ms=9)
    axE.set_xticks(x); axE.set_xticklabels([f"{c}" for c in cores])
    axE.set_xlabel("cores per move")
    axE.set_ylabel("absolute Elo  (bars = 90% confidence)")
    axE.set_title("…but strength stays flat within noise")
    axE.set_ylim(1700, 2800)
    _style(axE)
    fig.suptitle("Cores buy search, not strength — the small net is judge-limited, not search-limited",
                 fontsize=16, fontweight="bold", color=FG, y=1.02)
    _save(fig, "cores_scaling.png")


if __name__ == "__main__":
    loss_curves()
    elo_vs_timecontrol()
    small_net_vs_time()
    catchup_slopes()
    cores_scaling()
    print("all figures written to docs/images/")
