"""
EXPERIMENT 1 -- ground truth.

(a) Reproduce a real, whole-model CUDA initialization bit-for-bit on CPU.
(b) Show which tensors carry any device information at all.
(c) Validate the reproduction across many launch-grid values on real hardware.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT.joinpath('results').mkdir(exist_ok=True)
ROOT.joinpath('figures').mkdir(exist_ok=True)

import numpy as np
import torch

sys.path.insert(0, ".")
from philox import DEVICES, torch_uniform_layout
from model import ToyConfig, ToyGPT, state_dict_emulated, param_plan

cfg = ToyConfig()
OUT = {}


def cmp(name, a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    nbad = int((a != b).sum())
    return dict(name=name, numel=int(a.size), exact=int(a.size - nbad),
                maxdiff=float(np.abs(a - b).max()))


if __name__ == "__main__":
    dev = DEVICES["rtx4070tis"]
    seed = 424242

    # ---------- (a) whole model ----------
    m = ToyGPT(cfg).cuda()
    m.init_weights_on_device(seed)
    gpu = {k: v.detach().cpu().numpy() for k, v in m.named_param_dict().items()}
    emu = state_dict_emulated(cfg, dev, seed)

    rows = [cmp(k, gpu[k], emu[k]) for k in gpu]
    total = sum(r["numel"] for r in rows)
    exact = sum(r["exact"] for r in rows)
    print(f"whole model: {exact:,}/{total:,} parameters reproduced bit-exactly on CPU "
          f"({100*exact/total:.4f}%)")
    assert exact == total, "emulation mismatch"
    OUT["whole_model"] = dict(total=total, exact=exact,
                              n_tensors=len(rows), seed=seed,
                              profile=repr(dev))
    OUT["tensors"] = rows

    # ---------- (b) which tensors carry device information ----------
    other = {name: state_dict_emulated(cfg, DEVICES[name], seed) for name in ["h100", "gb200", "b300"]}
    print("\nper-tensor sensitivity to the launch grid:")
    print(f"{'tensor':38s} {'numel':>10s} {'grid':>5s}  same_as_GB200  same_as_H100  same_as_B300")
    sens = []
    for p in param_plan(cfg):
        name = p["name"]
        n = int(np.prod(p["shape"]))
        g = dev.grid_for(n)
        a = emu[name].ravel()
        s_gb = bool(np.array_equal(a, other["gb200"][name].ravel()))
        s_h = bool(np.array_equal(a, other["h100"][name].ravel()))
        s_b3 = bool(np.array_equal(a, other["b300"][name].ravel()))
        sens.append(dict(name=name, numel=n, grid=g, same_gb200=s_gb, same_h100=s_h, same_b300=s_b3))
        print(f"{name:38s} {n:>10,} {g:>5}  {str(s_gb):>9s}      {str(s_h):>9s}    {str(s_b3):>9s}")
    OUT["sensitivity"] = sens
    n_insens = sum(1 for s in sens if s["same_gb200"] and s["same_h100"] and s["same_b300"])
    par_insens = sum(s["numel"] for s in sens if s["same_gb200"] and s["same_h100"] and s["same_b300"])
    print(f"\n{n_insens}/{len(sens)} tensors ({par_insens:,} params, "
          f"{100*par_insens/total:.1f}%) are IDENTICAL on every device profile.")
    OUT["device_invariant"] = dict(tensors=n_insens, total_tensors=len(sens),
                                   params=par_insens, total_params=total)

    # ---------- (c) grid sweep on real hardware ----------
    print("\nvalidating the reproduction across launch grids on the real GPU:")
    grids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 80, 96, 112,
             128, 160, 192, 224, 256, 288, 320, 352, 384, 396]
    sweep = []
    ok = 0
    for g in grids:
        n = g * 256
        torch.manual_seed(1000 + g)
        a = torch.empty(n, device="cuda").uniform_(0, 1).cpu().numpy()
        b, _ = torch_uniform_layout(n, dev, 1000 + g, 0)
        good = bool(np.array_equal(a, b))
        ok += good
        sweep.append(dict(grid=g, numel=n, exact=good))
    print(f"  {ok}/{len(grids)} grid values reproduced bit-exactly "
          f"(grids {grids[0]}..{grids[-1]}, numel {grids[0]*256:,}..{grids[-1]*256:,})")
    OUT["grid_sweep"] = dict(ok=ok, total=len(grids), grids=grids)

    # large, capped tensors, multiple round-trips of the grid-stride loop
    big = []
    for n in [500_000, 1_000_000, 2_000_000, 8_000_000]:
        torch.manual_seed(77)
        a = torch.empty(n, device="cuda").uniform_(0, 1).cpu().numpy()
        b, _ = torch_uniform_layout(n, dev, 77, 0)
        good = bool(np.array_equal(a, b))
        big.append(dict(numel=n, grid=dev.grid_for(n), exact=good))
        print(f"  n={n:,} grid={dev.grid_for(n)} exact={good}")
    OUT["large_tensors"] = big

    with open(str(ROOT / "results") + "/exp1_validate.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nwrote results/exp1_validate.json")
