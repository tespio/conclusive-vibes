"""Full-stack validation: reproduce torch CUDA uniform_() exactly for many sizes."""
import sys
import pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from philox import DEVICES, torch_uniform_layout

dev = DEVICES["rtx4070tis"]
print("device:", dev)


def gpu_seq(seed, sizes):
    torch.manual_seed(seed)
    return [torch.empty(n, device="cuda").uniform_(0, 1).cpu().numpy() for n in sizes]


def sim_seq(seed, sizes):
    off = 0
    out = []
    for n in sizes:
        v, off = torch_uniform_layout(n, dev, seed, off)
        out.append(v)
    return out


if __name__ == "__main__":
    seed = 12345
    sizes = [1024, 100_000, 200_000, 405_504, 405_505, 1_000_000, 3_000_000]
    g = gpu_seq(seed, sizes)
    s = sim_seq(seed, sizes)
    for n, a, b in zip(sizes, g, s):
        nbad = int((a != b).sum())
        print(f"n={n:>9,} grid={dev.grid_for(n):>4} exact={n - nbad:>9,}/{n:<9,} maxdiff={np.abs(a-b).max():.3e}")

    # sequential offsets: one model init calls uniform_() dozens of times
    sizes2 = [4096, 65536, 1_048_576, 777, 2_500_000, 131072]
    g2 = gpu_seq(999, sizes2)
    s2 = sim_seq(999, sizes2)
    print("\nsequential tensors (offset accumulation):")
    off = 0
    for n, a, b in zip(sizes2, g2, s2):
        print(f"  n={n:>9,} offset={off:>9} exact={(a == b).all()}  grid={dev.grid_for(n)}")
        off += dev.counter_offset(n)
