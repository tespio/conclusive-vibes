"""
Bit-exact host-side reproduction of PyTorch's CUDA `Tensor.uniform_()` stream.

PyTorch's CUDA uniform/normal kernels are grid-stride loops over a Philox4x32-10
counter-based generator (ATen/native/cuda/DistributionTemplates.h):

    idx = blockIdx.x * blockDim.x + threadIdx.x
    curand_init(seed, idx, offset, &state)
    ... curand_uniform4(&state) ...

Each thread advances the generator by 4 words per outer-loop iteration and writes
the 4 returned floats to positions `blockDim.x * gridDim.x` apart.

Everything here is integer arithmetic plus IEEE-754 float32 multiply/add, so the
result is device-independent *given the launch grid*.  That is the whole point.
"""
from __future__ import annotations

import numpy as np

PH_M0 = np.uint32(0xD2511F53)
PH_M1 = np.uint32(0xCD9E8D57)
PH_W0 = np.uint32(0x9E3779B9)
PH_W1 = np.uint32(0xBB67AE85)

# curand_kernel.h float literals, rounded to float32 exactly as nvcc does
_2POW32_INV = np.float32(2.3283064e-10)
_2POW32_INV_2 = np.float32(2.3283064e-10 / 2.0)


def philox4x32_10(ctr: np.ndarray, key0, key1) -> np.ndarray:
    """ctr: (...,4) uint32; key0/key1 broadcastable to ctr.shape[:-1] -> (...,4)."""
    shape = np.broadcast_shapes(ctr.shape[:-1], np.shape(key0), np.shape(key1))
    c0 = np.broadcast_to(ctr[..., 0], shape).astype(np.uint32).copy()
    c1 = np.broadcast_to(ctr[..., 1], shape).astype(np.uint32).copy()
    c2 = np.broadcast_to(ctr[..., 2], shape).astype(np.uint32).copy()
    c3 = np.broadcast_to(ctr[..., 3], shape).astype(np.uint32).copy()
    k0 = np.broadcast_to(np.asarray(key0, dtype=np.uint32), shape).copy()
    k1 = np.broadcast_to(np.asarray(key1, dtype=np.uint32), shape).copy()
    with np.errstate(over="ignore"):
        for _ in range(10):
            p0 = c0.astype(np.uint64) * np.uint64(PH_M0)
            p1 = c2.astype(np.uint64) * np.uint64(PH_M1)
            hi0 = (p0 >> np.uint64(32)).astype(np.uint32)
            lo0 = p0.astype(np.uint32)
            hi1 = (p1 >> np.uint64(32)).astype(np.uint32)
            lo1 = p1.astype(np.uint32)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            k0 = np.uint32(k0 + PH_W0)
            k1 = np.uint32(k1 + PH_W1)
    return np.stack([c0, c1, c2, c3], axis=-1)


def curand_uniform(words: np.ndarray) -> np.ndarray:
    """curand's _curand_uniform: (0,1] float32."""
    return (words.astype(np.uint32).astype(np.float32) * _2POW32_INV + _2POW32_INV_2).astype(np.float32)


def curand4_uniform(seed, subseq, offset_words) -> np.ndarray:
    """(...,4) float32, one curand_uniform4() call per broadcast triple.

    NVIDIA curand's Philox state for curand_init(seed, subsequence, offset):
        key = (seed_lo, seed_hi)
        ctr = (offset/4, 0, subsequence_lo, subsequence_hi)
    Validated bit-exactly against sm_89 hardware (probes/probe_rng.py, exp1_validate.py).
    """
    seed = np.asarray(seed, dtype=np.uint64)
    subseq = np.asarray(subseq, dtype=np.uint64)
    off = np.asarray(offset_words, dtype=np.uint64)
    shape = np.broadcast_shapes(seed.shape, subseq.shape, off.shape)
    seed = np.broadcast_to(seed, shape)
    subseq = np.broadcast_to(subseq, shape)
    off = np.broadcast_to(off, shape)

    ctr = np.empty(shape + (4,), dtype=np.uint32)
    ctr[..., 0] = (off // 4).astype(np.uint32)
    ctr[..., 1] = 0
    ctr[..., 2] = (subseq & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    ctr[..., 3] = (subseq >> np.uint64(32)).astype(np.uint32)
    return curand_uniform(philox4x32_10(ctr, seed & 0xFFFFFFFF, seed >> 32))


# --------------------------------------------------------------------------------------
# PyTorch launch policy (calc_execution_policy).  The ONLY place hardware enters.
# --------------------------------------------------------------------------------------
BLOCK_SIZE = 256
UNROLL = 4  # sizeof(float4)/sizeof(float)


class DeviceProfile:
    def __init__(self, name: str, sm_count: int, max_threads_per_sm: int):
        self.name = name
        self.sm_count = int(sm_count)
        self.max_threads_per_sm = int(max_threads_per_sm)

    @property
    def blocks_per_sm(self) -> int:
        return self.max_threads_per_sm // BLOCK_SIZE

    @property
    def grid_cap(self) -> int:
        return self.sm_count * self.blocks_per_sm

    def grid_for(self, numel: int) -> int:
        return min((numel + BLOCK_SIZE - 1) // BLOCK_SIZE, self.grid_cap)

    def counter_offset(self, numel: int) -> int:
        return ((numel - 1) // (BLOCK_SIZE * self.grid_for(numel) * UNROLL) + 1) * 4

    def __repr__(self) -> str:
        return f"{self.name}(sm={self.sm_count}, thr/sm={self.max_threads_per_sm}, cap={self.grid_cap})"


DEVICES = {
    "h100": DeviceProfile("H100 SXM", 132, 2048),
    "h200": DeviceProfile("H200", 132, 2048),
    "b200": DeviceProfile("B200", 148, 2048),
    "gb200": DeviceProfile("GB200 (per GPU)", 148, 2048),
    "b300": DeviceProfile("B300/GB300", 152, 2048),
    "rtx4070tis": DeviceProfile("RTX 4070 Ti SUPER", 66, 1536),
    "a100": DeviceProfile("A100", 108, 2048),
    "mig_1g": DeviceProfile("H100 MIG 1g slice", 16, 2048),
}


def torch_uniform_layout(numel: int, dev: DeviceProfile, seed: int, offset_words: int):
    """Reproduce torch.empty(numel, device='cuda').uniform_(0,1) bit-exactly.

    Returns (values float32[numel], next_offset_words).
    """
    grid = dev.grid_for(numel)
    nthreads = grid * BLOCK_SIZE
    stride = nthreads * UNROLL
    p = np.arange(numel, dtype=np.int64)
    k = p // stride
    rem = p - k * stride
    ii = rem // nthreads
    idx = rem - ii * nthreads

    vals4 = curand4_uniform(seed, idx.astype(np.uint64), (offset_words + 4 * k).astype(np.uint64))
    out = vals4[np.arange(numel), ii]
    out[out == np.float32(1.0)] = np.float32(0.0)  # torch's reverse-bound fix
    return out.astype(np.float32), offset_words + dev.counter_offset(numel)
