"""Validate the host-side Philox reproduction against the real GPU."""
import sys
import pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from philox import philox4x32_10, curand_uniform, DEVICES, torch_uniform_layout


def gpu_uniform(seed: int, n: int) -> np.ndarray:
    torch.manual_seed(seed)
    return torch.empty(n, device="cuda").uniform_(0, 1).cpu().numpy()


def try_variant(seed, n_first, ctr_mode, key_mode):
    s = np.arange(n_first, dtype=np.uint64)
    ctr = np.zeros((n_first, 4), dtype=np.uint32)
    if ctr_mode == "off0_sub23":
        ctr[:, 2] = s
    elif ctr_mode == "sub0":
        ctr[:, 0] = s
    elif ctr_mode == "sub1":
        ctr[:, 1] = s
    elif ctr_mode == "sub01":
        ctr[:, 0] = s
        ctr[:, 1] = 0
    if key_mode == "lo_hi":
        k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    else:
        k0, k1 = (seed >> 32) & 0xFFFFFFFF, seed & 0xFFFFFFFF
    w = philox4x32_10(ctr, k0, k1)
    return curand_uniform(w)[:, 0]


if __name__ == "__main__":
    torch.manual_seed(0)
    seed = 12345
    g = gpu_uniform(seed, 64)
    print("GPU first 6:", g[:6])
    for cm in ["off0_sub23", "sub0", "sub1", "sub01"]:
        for km in ["lo_hi", "hi_lo"]:
            v = try_variant(seed, 64, cm, km)
            print(f"{cm:12s} {km:5s} matches={int((v == g).sum())}/64  first={v[0]:.9g}")
