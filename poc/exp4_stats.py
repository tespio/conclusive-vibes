"""
EXPERIMENT 4 -- the statistics.

His headline numbers are "8 sigma of accidental match across six trillion
combinations" and "real matches from 49 to 1,385 sigma", alongside a
"0.999995 correlation".

Those numbers cannot all describe the same thing.  Under the Gaussian null he
himself invokes, significance z = r*sqrt(n)/sqrt(1-r^2) grows with the number of
samples n, so a correlation of 0.999995 over a billion weights is ~1e7 sigma,
not 1,385.  Conversely 1,385 sigma at r=0.999995 implies n_eff ~ 19 samples.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROOT.joinpath("results").mkdir(exist_ok=True)
ROOT.joinpath("figures").mkdir(exist_ok=True)

OUT = {}


def z_of_r(r: float, n: float) -> float:
    return r * math.sqrt(n) / math.sqrt(max(1e-300, 1 - r * r))


def r_of_z(z: float, n: float) -> float:
    return z / math.sqrt(n + z * z)


def expected_max_z(k: float) -> float:
    """Expected maximum of k iid standard normals (asymptotic)."""
    if k < 2:
        return 0.0
    a = math.sqrt(2 * math.log(k))
    return a - (math.log(math.log(k)) + math.log(4 * math.pi)) / (2 * a)


if __name__ == "__main__":
    print("A. what the claimed sigmas must mean")
    print(f"{'claim':>28s} {'implied r at N=1e9':>20s} {'implied r at N=1e12':>22s}")
    tbl = []
    for z in [8, 20, 49, 200, 1385]:
        r9, r12 = r_of_z(z, 1e9), r_of_z(z, 1e12)
        tbl.append(dict(z=z, r_at_1e9=r9, r_at_1e12=r12))
        print(f"{z:>20,} sigma {r9:>20.6f} {r12:>22.8f}")

    print("\nB. what the 0.999995 correlation must mean")
    tbl2 = []
    for n in [19, 1e6, 4.2e6, 1e9, 1e12]:
        z = z_of_r(0.999995, n)
        tbl2.append(dict(n=n, z=z))
        print(f"  r=0.999995 over {n:>12,.0f} weights -> {z:>14,.0f} sigma")
    r = 0.999995
    n_implied = 1385 ** 2 * (1 - r * r) / (r * r)
    print(f"  his 1,385 sigma at r=0.999995 implies n_eff = {n_implied:.1f} independent samples")

    print("\nC. multiple comparisons: the '8 sigma' noise floor")
    mc = {}
    rng = np.random.default_rng(0)
    for k in [10, 1_000, 100_000, 10_000_000]:
        mx = float(rng.standard_normal(k).max())
        mc[k] = dict(mc=mx, theory=expected_max_z(k))
        print(f"  K={k:>12,}  observed max {mx:6.3f}   E[max] ~ {expected_max_z(k):6.3f}")
    K = 6e12
    print(f"  K=6e12 (his search):  E[max] ~ {expected_max_z(K):.3f} sigma  <- his 'about 8'")
    OUT["multiple_comparisons"] = dict(mc=mc, K=K, theory=expected_max_z(K))

    print("\nD. the same statistic on a model that never saw Blackwell")
    with open(str(ROOT / "results") + "/exp3_train.json") as f:
        tr = json.load(f)
    band = []
    for run in ["full", "partial"]:
        for h in tr[run]["history"]:
            if h["sigma"]:
                band.append(dict(run=run, step=h["step"], cos=h["cos"], sigma=h["sigma"],
                                 exact_frac=h["exact_frac"], detector=h["detect_matches"]))
    print(f"{'step':>6} {'cos':>12} {'sigma':>12} {'exact %':>9} {'detector':>10}")
    for b in band:
        if b["run"] == "full":
            print(f"{b['step']:>6} {b['cos']:>12.7f} {b['sigma']:>12,.0f} "
                  f"{100*b['exact_frac']:>8.3f}% {b['detector']:>6}/256")
    print("  his entire reported range [49, 1385] is BELOW every value here,")
    print("  on a model initialised on an RTX 4070 Ti SUPER and never touched by Blackwell.")

    OUT["claims"] = dict(z_table=tbl, r_table=tbl2, n_eff_at_1385=n_implied)
    OUT["sigma_vs_step"] = band
    with open(str(ROOT / "results") + "/exp4_stats.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nwrote results/exp4_stats.json")
