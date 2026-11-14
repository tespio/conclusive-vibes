"""
A small GPT-style model whose *entire* initialization can be reproduced
bit-for-bit either on a real CUDA GPU or in pure NumPy (no GPU).

The point is not the model.  The point is that `init_weights` consumes the
CUDA generator in a fixed, documented order, so its output is a pure function
of (seed, tensor shapes, device launch-grid).  The launch grid is two integers
reported by the driver.  Nothing else about the hardware enters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from philox import DeviceProfile, torch_uniform_layout


@dataclass
class ToyConfig:
    vocab: int = 4096
    d: int = 384
    n_layer: int = 4
    n_head: int = 6
    mlp_mult: int = 4
    seq_len: int = 64

    @property
    def mlp_dim(self) -> int:
        return self.d * self.mlp_mult


def param_plan(cfg: ToyConfig) -> list[dict]:
    """Ordered description of every parameter, matching ToyGPT.init_weights()."""
    plan: list[dict] = []

    def uni(name, shape, fan_in):
        plan.append(dict(name=name, shape=tuple(shape), kind="uniform",
                         bound=1.0 / math.sqrt(fan_in)))

    def const(name, shape, val):
        plan.append(dict(name=name, shape=tuple(shape), kind="const", val=val))

    uni("tok_emb.weight", (cfg.vocab, cfg.d), cfg.d)
    for i in range(cfg.n_layer):
        const(f"blocks.{i}.ln1.weight", (cfg.d,), 1.0)
        const(f"blocks.{i}.ln1.bias", (cfg.d,), 0.0)
        uni(f"blocks.{i}.attn.qkv.weight", (3 * cfg.d, cfg.d), cfg.d)
        uni(f"blocks.{i}.attn.qkv.bias", (3 * cfg.d,), cfg.d)
        uni(f"blocks.{i}.attn.proj.weight", (cfg.d, cfg.d), cfg.d)
        uni(f"blocks.{i}.attn.proj.bias", (cfg.d,), cfg.d)
        const(f"blocks.{i}.ln2.weight", (cfg.d,), 1.0)
        const(f"blocks.{i}.ln2.bias", (cfg.d,), 0.0)
        uni(f"blocks.{i}.mlp.fc1.weight", (cfg.mlp_dim, cfg.d), cfg.d)
        uni(f"blocks.{i}.mlp.fc1.bias", (cfg.mlp_dim,), cfg.d)
        uni(f"blocks.{i}.mlp.fc2.weight", (cfg.d, cfg.mlp_dim), cfg.mlp_dim)
        uni(f"blocks.{i}.mlp.fc2.bias", (cfg.d,), cfg.mlp_dim)
    const("ln_f.weight", (cfg.d,), 1.0)
    const("ln_f.bias", (cfg.d,), 0.0)
    uni("lm_head.weight", (cfg.vocab, cfg.d), cfg.d)
    return plan


def apply_uniform(u: np.ndarray, bound: float) -> np.ndarray:
    """Reproduce torch's uniform_(-bound, bound) float32 transform exactly.

    torch compiles `rand * range + from` with FMA contraction, so the host
    reproduction must fuse it too: float64 holds the exact product of two
    float32 values, so rounding once at the end matches the FMA.
    """
    lo = np.float32(-bound)
    hi = np.float32(bound)
    rng = np.float32(hi - lo)
    v = (u.astype(np.float64) * np.float64(rng) + np.float64(lo)).astype(np.float32)
    v = np.where(v == hi, lo, v).astype(np.float32)
    return v


def state_dict_emulated(cfg: ToyConfig, profile: DeviceProfile, seed: int,
                        plan: list[dict] | None = None) -> dict[str, np.ndarray]:
    """Generate the full initialization with no GPU, for any launch grid."""
    plan = plan or param_plan(cfg)
    off = 0
    out: dict[str, np.ndarray] = {}
    for p in plan:
        n = int(np.prod(p["shape"]))
        if p["kind"] == "uniform":
            u, off = torch_uniform_layout(n, profile, seed, off)
            v = apply_uniform(u, p["bound"])
        else:
            v = np.full(n, p["val"], dtype=np.float32)
        out[p["name"]] = v.reshape(p["shape"])
    return out


class ToyGPT(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layer):
            blk = nn.ModuleDict({
                "ln1": nn.LayerNorm(cfg.d),
                "attn": nn.ModuleDict({
                    "qkv": nn.Linear(cfg.d, 3 * cfg.d),
                    "proj": nn.Linear(cfg.d, cfg.d),
                }),
                "ln2": nn.LayerNorm(cfg.d),
                "mlp": nn.ModuleDict({
                    "fc1": nn.Linear(cfg.d, cfg.mlp_dim),
                    "fc2": nn.Linear(cfg.mlp_dim, cfg.d),
                }),
            })
            self.blocks.append(blk)
        self.ln_f = nn.LayerNorm(cfg.d)
        self.lm_head = nn.Embedding(cfg.vocab, cfg.d)

    def named_param_dict(self) -> dict[str, nn.Parameter]:
        return dict(self.named_parameters())

    def init_weights_on_device(self, seed: int) -> None:
        """Run the canonical init *on whatever device the params live on*."""
        torch.manual_seed(seed)
        pd = self.named_param_dict()
        with torch.no_grad():
            for p in param_plan(self.cfg):
                t = pd[p["name"]]
                if p["kind"] == "uniform":
                    t.uniform_(-p["bound"], p["bound"])
                else:
                    t.fill_(p["val"])

    def load_numpy(self, arrays: dict[str, np.ndarray]) -> None:
        pd = self.named_param_dict()
        with torch.no_grad():
            for name, arr in arrays.items():
                pd[name].copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(pd[name].device))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T = idx.shape
        x = self.tok_emb(idx)
        for blk in self.blocks:
            h = blk["ln1"](x)
            qkv = blk["attn"]["qkv"](h)
            q, k, v = qkv.split(cfg.d, dim=-1)
            q = q.view(B, T, cfg.n_head, cfg.d // cfg.n_head).transpose(1, 2)
            k = k.view(B, T, cfg.n_head, cfg.d // cfg.n_head).transpose(1, 2)
            v = v.view(B, T, cfg.n_head, cfg.d // cfg.n_head).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).reshape(B, T, cfg.d)
            x = x + blk["attn"]["proj"](att)
            h = blk["ln2"](x)
            x = x + blk["mlp"]["fc2"](F.gelu(blk["mlp"]["fc1"](h)))
        x = self.ln_f(x)
        return x @ self.lm_head.weight.t()
