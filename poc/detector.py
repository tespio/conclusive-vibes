"""
A faithful re-implementation of the claimed "checkpoint forensics" method:

    given a model checkpoint, recover (seed, GPU launch-grid) by brute force
    and declare the GPU family that initialised it.

This is not a strawman.  On a step-0 checkpoint it works perfectly, because the
initialization stream is an exact deterministic function of (seed, shapes, grid).
That is precisely why a match says nothing about where the job ran.
"""
from __future__ import annotations

import time

import numpy as np

from philox import BLOCK_SIZE, UNROLL, DeviceProfile, curand4_uniform
from model import ToyConfig, apply_uniform, param_plan


def probe_values(profile: DeviceProfile, seeds: np.ndarray, positions: np.ndarray,
                 offset: int, numel: int) -> np.ndarray:
    """Values at `positions` for each seed: shape (len(seeds), len(positions))."""
    seeds = np.asarray(seeds, dtype=np.uint64)[:, None]
    pos = np.asarray(positions, dtype=np.int64)
    grid = profile.grid_for(numel)
    nthreads = grid * BLOCK_SIZE
    stride = nthreads * UNROLL

    k = pos // stride
    rem = pos - k * stride
    ii = rem // nthreads
    idx = rem - ii * nthreads

    vals4 = curand4_uniform(seeds, idx[None, :].astype(np.uint64),
                            (offset + 4 * k)[None, :].astype(np.uint64))
    return vals4[:, np.arange(pos.size), ii]


class Detector:
    """Brute-force (seed, launch-grid) recovery from a checkpoint's first big tensor."""

    def __init__(self, cfg: ToyConfig, profiles: list[DeviceProfile],
                 n_probe: int = 256, probe_start: int = 1_000_000):
        self.cfg = cfg
        self.profiles = profiles
        self.n_probe = n_probe
        self.probe_start = probe_start
        self.plan = param_plan(cfg)
        self.target = None
        for p in self.plan:
            n = int(np.prod(p["shape"]))
            if p["kind"] == "uniform" and n > 400_000:
                self.target = (p["name"], n, p["bound"])
                break
        if self.target is None:
            raise RuntimeError("no large uniform tensor to probe")

    def _offset_for(self, profile: DeviceProfile) -> int:
        off = 0
        for p in self.plan:
            if p["name"] == self.target[0]:
                return off
            if p["kind"] == "uniform":
                off += profile.counter_offset(int(np.prod(p["shape"])))
        return off

    def detect(self, ckpt: dict[str, np.ndarray], seeds: np.ndarray,
               chunk: int = 4096, verbose: bool = False) -> dict:
        name, numel, bound = self.target
        start = min(self.probe_start, numel - self.n_probe)
        pos = start + np.arange(self.n_probe, dtype=np.int64)
        want = ckpt[name].ravel()[pos]

        t0 = time.time()
        n_cand = 0
        best = dict(profile=None, seed=None, matches=-1)
        per_profile: dict[str, dict] = {}
        for prof in self.profiles:
            off = self._offset_for(prof)
            pb = (-1, None)
            for i in range(0, len(seeds), chunk):
                s = seeds[i:i + chunk]
                vals = apply_uniform(probe_values(prof, s, pos, off, numel), bound)
                m = (vals == want[None, :]).sum(axis=1)
                j = int(np.argmax(m))
                if int(m[j]) > pb[0]:
                    pb = (int(m[j]), int(s[j]))
                n_cand += len(s)
            per_profile[prof.name] = dict(best_matches=pb[0], best_seed=pb[1])
            if verbose:
                print(f"    {prof.name:24s} best {pb[0]:>4}/{self.n_probe}  seed {pb[1]}")
            if pb[0] > best["matches"]:
                best = dict(profile=prof.name, seed=pb[1], matches=pb[0])
        dt = time.time() - t0
        return dict(best=best, n_probe=self.n_probe, n_candidates=n_cand,
                    elapsed_s=dt, candidates_per_s=n_cand / max(dt, 1e-9),
                    per_profile=per_profile,
                    conclusive=best["matches"] == self.n_probe,
                    tensor=name)
