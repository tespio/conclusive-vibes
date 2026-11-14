"""
EXPERIMENT 2 -- the forgery.

A checkpoint whose initialization is *bit-identical* to what a GB200 would
produce is generated on this laptop's CPU, with no Blackwell (and no GPU at all)
anywhere in the loop.  The claimed forensic method then "conclusively" identifies
a GB200.

Ground truth: the only GPU in this machine is an RTX 4070 Ti SUPER.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT.joinpath('results').mkdir(exist_ok=True)
ROOT.joinpath('figures').mkdir(exist_ok=True)
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from detector import Detector
from model import ToyConfig, ToyGPT, param_plan, state_dict_emulated
from philox import DEVICES

cfg = ToyConfig()
SEED = 424242
PROFILES = [DEVICES[n] for n in ["rtx4070tis", "h100", "gb200", "b300"]]
OUT = {}


def digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for p in param_plan(cfg):
        h.update(np.ascontiguousarray(arrays[p["name"]]).tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    det = Detector(cfg, PROFILES)
    print(f"target tensor: {det.target[0]} ({det.target[1]:,} params)\n")

    # ---------- ground truth: a real init on the real GPU ----------
    m = ToyGPT(cfg).cuda()
    m.init_weights_on_device(SEED)
    gpu_ckpt = {k: v.detach().cpu().numpy() for k, v in m.named_param_dict().items()}
    print("[A] real CUDA init on RTX 4070 Ti SUPER (the only GPU present)")

    # ---------- forgery: GB200 layout, generated on the CPU ----------
    t0 = time.time()
    forged = state_dict_emulated(cfg, DEVICES["gb200"], SEED)
    t_gen = time.time() - t0
    n_params = sum(a.size for a in forged.values())
    print(f"[B] forged 'GB200' init generated on CPU: {n_params:,} params in {t_gen:.2f}s")

    # the forged stream differs from the real one, so this is a real layout change
    a = forged["tok_emb.weight"].ravel()
    b = gpu_ckpt["tok_emb.weight"].ravel()
    same = int((a == b).sum())
    print(f"    forged vs real GPU stream: {same:,}/{a.size:,} identical "
          f"(differs as expected for a different launch grid)")

    # reproduce on GPU: our Ada card cannot produce this stream
    m.init_weights_on_device(SEED)
    print(f"    GPU re-init still produces its own stream: "
          f"{int((m.named_param_dict()['tok_emb.weight'].detach().cpu().numpy().ravel() == a).sum()):,}/"
          f"{a.size:,} matches to forged")

    # ---------- detector: his method, both checkpoints ----------
    seeds = np.arange(SEED - 20_000, SEED + 20_000, dtype=np.uint64)
    print(f"\n[detector] searching {len(seeds):,} seeds x {len(PROFILES)} device profiles ...")
    res_gpu = det.detect(gpu_ckpt, seeds, verbose=True)
    print(f"  real GPU checkpoint -> {res_gpu['best']}  conclusive={res_gpu['conclusive']}")
    res_forged = det.detect(forged, seeds, verbose=True)
    print(f"  forged checkpoint   -> {res_forged['best']}  conclusive={res_forged['conclusive']}")

    OUT["detector_gpu"] = res_gpu
    OUT["detector_forged"] = res_forged
    OUT["truth"] = "RTX 4070 Ti SUPER (66 SMs), zero Blackwells present"
    OUT["forgery"] = dict(n_params=n_params, gen_seconds=t_gen,
                          digest=digest(forged),
                          digest_gpu=digest(gpu_ckpt),
                          identical_to_gpu_stream=same)

    # ---------- the same trick for other device profiles ----------
    print("\n[forgeries] CPU-generated checkpoints for other claimed devices:")
    variants = {}
    for name in ["h100", "b300"]:
        t0 = time.time()
        ck = state_dict_emulated(cfg, DEVICES[name], SEED)
        g = time.time() - t0
        r = det.detect(ck, seeds)
        variants[name] = dict(gen_seconds=g, verdict=r["best"], conclusive=r["conclusive"])
        print(f"  forged {DEVICES[name].name:22s} -> {r['best']}  conclusive={r['conclusive']}")
    OUT["forged_variants"] = variants

    # ---------- scale: what does the 'smoking gun' cost? ----------
    print("\n[scale] generating a 1-trillion-parameter init in the GB200 layout on CPU:")
    big_n = 32_000_000
    t0 = time.time()
    _ = state_dict_emulated
    from philox import torch_uniform_layout
    _v, _ = torch_uniform_layout(big_n, DEVICES["gb200"], 953049583, 0)
    dt = time.time() - t0
    rate = big_n / dt
    print(f"    {big_n:,} values in {dt:.2f}s -> {rate/1e6:.1f}M values/s")
    for n in [1e9, 1e12]:
        secs = n / rate
        print(f"    {n:.0e} params: ~{secs/60:.1f} min  (~{n*4/1e9:.0f} GB fp32)")
    OUT["scale"] = dict(values_per_s=rate, big_n=big_n, seconds=dt)

    # 8-byte hash of the 'DeepSeek seed' stream under the GB200 layout
    h = hashlib.sha256(_v.tobytes()).hexdigest()
    print(f"    sha256 of first {big_n:,} values (seed 953049583, GB200 grid): {h[:32]}...")
    OUT["deepseek_seed_demo"] = dict(seed=953049583, grid=DEVICES["gb200"].grid_for(big_n),
                                     sha256=h, n=big_n)

    np.savez(str(ROOT / "results") + "/forged_gb200_step0.npz",
             **{k: v for k, v in forged.items()})
    with open(str(ROOT / "results") + "/exp2_forge.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nwrote results/exp2_forge.json and results/forged_gb200_step0.npz")
