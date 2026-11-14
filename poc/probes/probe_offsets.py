import sys
import pathlib
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import ToyConfig, ToyGPT, param_plan
from philox import DEVICES

cfg = ToyConfig()
dev = DEVICES["rtx4070tis"]
m = ToyGPT(cfg).cuda()
pd = m.named_param_dict()
torch.manual_seed(424242)


def state():
    b = torch.cuda.get_rng_state().numpy()
    return int.from_bytes(b[:8].tobytes(), "little"), int.from_bytes(b[8:16].tobytes(), "little")


print("initial state (seed, offset):", state())
prev = state()[1]
for i, p in enumerate(param_plan(cfg)):
    t = pd[p["name"]]
    with torch.no_grad():
        if p["kind"] == "uniform":
            t.uniform_(-p["bound"], p["bound"])
        else:
            t.fill_(p["val"])
    s = state()
    n = int(np.prod(p["shape"]))
    expect = dev.counter_offset(n)
    if i < 8 or s[1] - prev != expect:
        print(f"{i:2d} {p['name']:36s} numel={n:>8,} grid={dev.grid_for(n):4d} "
              f"delta={s[1]-prev:>6} expect={expect:>6} {'OK' if s[1]-prev==expect else 'MISMATCH'}")
    prev = s[1]
