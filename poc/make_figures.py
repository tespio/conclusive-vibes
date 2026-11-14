"""Figures for the report."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT.joinpath('results').mkdir(exist_ok=True)
ROOT.joinpath('figures').mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "text.color": "#111111", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.7,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})
RED, BLUE, GREY, ORANGE = "#c0392b", "#2c6fbb", "#7f8c8d", "#e67e22"


def main():
    ex1 = json.load(open(str(ROOT / "results") + "/exp1_validate.json"))
    ex2 = json.load(open(str(ROOT / "results") + "/exp2_forge.json"))
    ex3 = json.load(open(str(ROOT / "results") + "/exp3_train.json"))
    ex4 = json.load(open(str(ROOT / "results") + "/exp4_stats.json"))

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.2))

    # ---------------- A: misattribution matrix ----------------
    a = ax[0, 0]
    m = np.array([[256, 1], [1, 256]], dtype=float)
    a.imshow(m, cmap="Blues", vmin=0, vmax=256)
    a.set_xticks([0, 1], ["RTX 4070 Ti S\n(Ada)", "GB200\n(Blackwell)"])
    a.set_yticks([0, 1], ["real init\non the RTX 4070", "forged on CPU,\nlabelled GB200"])
    for i in range(2):
        for j in range(2):
            v = int(m[i, j])
            a.text(j, i, f"{v}/256", ha="center", va="center", fontsize=15,
                   fontweight="bold", color="white" if v > 128 else "#222222")
    a.set_title("A. The detector's verdict, by ground truth\n"
                "(256 = bit-for-bit, 'conclusive')")
    a.grid(False)

    # ---------------- B: exactness vs training ----------------
    b = ax[0, 1]
    full = ex3["full"]["history"]
    part = ex3["partial"]["history"]
    steps = [h["step"] for h in full]
    b.plot(steps, [100 * h["exact_frac"] for h in full], "o-", color=RED,
           label="full vocabulary coverage", lw=2, ms=5)
    b.plot([h["step"] for h in part], [100 * h["exact_frac"] for h in part], "s--",
           color=BLUE, label="64/4096 tokens seen (null-space rows keep init)", lw=1.8, ms=4)
    b.set_xscale("symlog", linthresh=1)
    b.set_xlabel("optimizer steps")
    b.set_ylabel("% of parameters bit-identical to step 0")
    b.set_title("B. One optimizer step destroys the bit-exact fingerprint")
    b.set_ylim(-2, 102)
    b.annotate("step 0: 100%\n(only here does he have 'bit-for-bit')",
               xy=(0, 100), xytext=(3, 74), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9))
    b.annotate("step 1: 88% gone;\nstep 2: detector fails",
               xy=(2, 9.3), xytext=(14, 30), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9))
    b.legend(loc="center right", fontsize=8)
    b.grid(alpha=0.35)

    # ---------------- C: sigma vs steps ----------------
    c = ax[1, 0]
    c.axhspan(49, 1385, color=ORANGE, alpha=0.22,
              label="his reported range (49–1,385σ)")
    c.plot(steps[1:], [h["sigma"] for h in full[1:]], "o-", color=RED, lw=2, ms=5,
           label="same statistic, RTX 4070-initialised model")
    c.set_xscale("symlog", linthresh=1)
    c.set_yscale("log")
    c.set_xlabel("optimizer steps")
    c.set_ylabel("reported 'sigma' of the match")
    c.set_title("C. The σ number measures distance from init, not hardware")
    c.annotate("step 600: 1.5×10$^4$σ — still 11×\nhis strongest claim",
               xy=(600, 1.54e4), xytext=(18, 1.6e3), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9))
    c.set_ylim(2e3, 2e6)
    c.legend(loc="upper right", fontsize=8)
    c.grid(alpha=0.35, which="both")

    # ---------------- D: sigma <-> correlation <-> N ----------------
    d = ax[1, 1]
    N = np.logspace(2, 13, 400)
    for z, col, lab in [(8, GREY, "8σ (his noise floor)"),
                        (49, ORANGE, "49σ"),
                        (1385, RED, "1,385σ")]:
        d.plot(N, z / np.sqrt(N + z * z), color=col, lw=2, label=lab)
    d.axhline(0.999995, color=BLUE, ls="--", lw=1.5)
    d.plot([19.2], [0.999995], "o", color=BLUE, ms=8, zorder=5)
    d.text(6e11, 1.9, "his 0.999995 correlation", color=BLUE, fontsize=8.5, ha="right")
    d.annotate("implies n$_{eff}$ ≈ 19, not 10$^9$", xy=(19.2, 0.999995), xytext=(120, 0.012),
               fontsize=8.5, color=BLUE,
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_xlabel("number of (assumed independent) weights compared, N")
    d.set_ylabel("correlation r implied by a given σ")
    d.set_title("D. 0.999995 and 1,385σ are mutually inconsistent")
    d.legend(loc="lower left", fontsize=8)
    d.grid(alpha=0.35, which="both")
    d.set_ylim(1e-4, 3.0)

    fig.suptitle("Checkpoint forensics, audited: a software artifact, not a hardware fingerprint",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(ROOT / "figures") + "/fig1_audit.png", dpi=170)
    print("wrote figures/fig1_audit.png")


if __name__ == "__main__":
    main()
