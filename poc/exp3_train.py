"""
EXPERIMENT 3 -- what training does to the fingerprint.

Initialise a real transformer on the real GPU, train it, and measure at each
checkpoint:
  * fraction of parameters still bit-identical to step 0
  * cosine similarity to step 0
  * the "N-sigma" statistic his report is built on
  * whether the detector can still recover (seed, device)

Two runs:
  full     -- every vocabulary token is seen, so every embedding row is trained
  partial  -- only 64 of 4096 tokens are seen; untouched embedding rows keep
              their step-0 values exactly (the "null space" his method needs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT.joinpath("results").mkdir(exist_ok=True)
ROOT.joinpath("figures").mkdir(exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from detector import Detector
from model import ToyConfig, ToyGPT
from philox import DEVICES

cfg = ToyConfig()
SEED = 424242
PROFILES = [DEVICES[n] for n in ["rtx4070tis", "h100", "gb200", "b300"]]
EVAL_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600]
OUT = {}


def make_batch(bs: int, vocab_used: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    half = cfg.seq_len // 2
    x = torch.randint(0, vocab_used, (bs, half), generator=gen, device="cuda")
    full = torch.cat([x, x], dim=1)
    return full[:, :-1], full[:, 1:]


def metrics(model: ToyGPT, W0: dict[str, torch.Tensor], step: int) -> dict:
    tot = exact = 0
    dot = n0 = n1 = 0.0
    per = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            a = p.detach().double().ravel()
            b = W0[name].detach().double().ravel()
            ex = int((p.detach().ravel() == W0[name].ravel()).sum())
            tot += a.numel()
            exact += ex
            dot += float((a * b).sum())
            n0 += float((a * a).sum())
            n1 += float((b * b).sum())
            per.append(dict(name=name, numel=int(a.numel()), exact=ex,
                            cos=float((a * b).sum() / (a.norm() * b.norm() + 1e-30))))
    cos = dot / (np.sqrt(n0) * np.sqrt(n1))
    n_eff_implied = None
    z = None
    if cos < 1.0:
        z = cos * np.sqrt(tot) / np.sqrt(max(1e-30, 1 - cos * cos))
    return dict(step=step, total=int(tot), exact=int(exact),
                exact_frac=exact / tot, cos=float(cos), sigma=float(z) if z else None,
                per_tensor=per)


def run(cover_all: bool, steps: int, tag: str) -> dict:
    vocab_used = cfg.vocab if cover_all else 64
    gen = torch.Generator(device="cuda").manual_seed(1234)
    model = ToyGPT(cfg).cuda()
    model.init_weights_on_device(SEED)
    W0 = {k: v.detach().clone() for k, v in model.named_param_dict().items()}

    det = Detector(cfg, PROFILES)
    seeds = np.arange(SEED - 2000, SEED + 2000, dtype=np.uint64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)

    hist = []
    loss_ema = None
    t0 = time.time()
    for step in range(steps + 1):
        if step in EVAL_STEPS or step == steps:
            m = metrics(model, W0, step)
            ck = {det.target[0]: model.named_param_dict()[det.target[0]].detach().cpu().numpy()}
            d = det.detect(ck, seeds)
            m["detect_matches"] = d["best"]["matches"]
            m["detect_profile"] = d["best"]["profile"]
            m["detect_seed"] = d["best"]["seed"]
            m["detect_conclusive"] = d["conclusive"]
            m["loss"] = loss_ema
            hist.append(m)
            print(f"  step {step:>4}  loss {loss_ema if loss_ema is None else round(loss_ema,3)!s:>7}"
                  f"  exact {100*m['exact_frac']:6.3f}%  cos {m['cos']:.6f}"
                  f"  sigma {('%.3g' % m['sigma']) if m['sigma'] else 'inf':>8}"
                  f"  detector {m['detect_matches']}/256 {m['detect_profile']}")
        if step == steps:
            break
        x, y = make_batch(32, vocab_used, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_ema = float(loss.detach()) if loss_ema is None else 0.95 * loss_ema + 0.05 * float(loss.detach())

    dt = time.time() - t0
    print(f"  {tag}: {steps} steps in {dt:.1f}s, final loss {loss_ema:.3f}")
    return dict(tag=tag, seed=SEED, steps=steps, vocab_used=vocab_used,
                truth_profile="RTX 4070 Ti SUPER", history=hist, seconds=dt)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    print("run 1: full vocabulary coverage (every embedding row is trained)")
    full = run(cover_all=True, steps=600, tag="full")
    print("\nrun 2: partial coverage (64 of 4096 tokens used -> untouched rows keep init)")
    part = run(cover_all=False, steps=600, tag="partial")

    # precision-cast calibration of the "0.999995 correlation" figure
    m = ToyGPT(cfg).cuda()
    m.init_weights_on_device(SEED)
    w = m.named_param_dict()["tok_emb.weight"].detach().double().ravel()
    cal = {}
    for name, cast in [("bf16", lambda t: t.bfloat16().float()),
                       ("fp16", lambda t: t.half().float()),
                       ("int8", lambda t: (t / t.abs().max() * 127).round().clamp(-127, 127) / 127 * t.abs().max())]:
        q = cast(w.float()).double()
        cal[name] = float((q * w).sum() / (q.norm() * w.norm()))
    OUT["cast_calibration"] = cal
    print("\ncorrelation of a step-0 tensor with its own precision cast:")
    for k, v in cal.items():
        print(f"  {k:5s} r = {v:.9f}")

    OUT["full"] = full
    OUT["partial"] = part
    with open(str(ROOT / "results") + "/exp3_train.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nwrote results/exp3_train.json")
