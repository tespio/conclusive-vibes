"""Can we reproduce a real nn.Linear's kaiming-uniform init with no GPU?"""
import math
import sys
import pathlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from philox import DEVICES, torch_uniform_layout

dev = DEVICES["rtx4070tis"]


def torch_uniform_vals(n, seed, off, lo, hi):
    u, nxt = torch_uniform_layout(n, dev, seed, off)
    lo32, hi32 = np.float32(lo), np.float32(hi)
    rng32 = np.float32(hi32 - lo32)
    v = (u * rng32 + lo32).astype(np.float32)
    v = np.where(v == hi32, lo32, v).astype(np.float32)
    return v, nxt


if __name__ == "__main__":
    lin = nn.Linear(4096, 1024, bias=False).cuda()
    torch.manual_seed(7)
    lin.reset_parameters()                            # init happens *on* the GPU
    w_gpu = lin.weight.detach().cpu().numpy().ravel()

    fan_in = 4096
    bound = 1.0 / math.sqrt(fan_in)
    # torch also initialises bias (None here); stream starts at offset 0
    v, off = torch_uniform_vals(4096 * 1024, 7, 0, -bound, bound)
    nbad = int((v != w_gpu).sum())
    print(f"nn.Linear(4096,1024) kaiming_uniform: exact {len(v)-nbad}/{len(v)}  maxdiff={np.abs(v-w_gpu).max():.3e}")
    print("gpu[0:4]", w_gpu[:4])
    print("sim[0:4]", v[:4])
    print("next offset", off, "grid", dev.grid_for(4096 * 1024))
