# We ran the checkpoint-forensics repo

The repo is real: [macrogrants/checkpoint-forensics](https://github.com/macrogrants/checkpoint-forensics).
Pinned Hugging Face rows with sha256 checks, a Philox4x32-10 kernel with Random123 test
vectors, MIT licence. We cloned it and ran `python replay.py` on this machine. It is CPU-only,
so the GPU is irrelevant.

It reproduces his table exactly:

| model | seed | SMs | rows | median r | seed+1 | next best |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3-Base | 953049583 | 132 | 8/8 | 0.999999 | −0.0038 | 183 (0.016) |
| DeepSeek-V4-Flash | 953049583 | 148 | 461/469 | 0.999999 | 0.0016 | 209 (0.023) |
| Kimi-K3 | 20256586 | 152 | 13/13 | 0.999996 | 0.0042 | 69 (0.018) |
| MiniMax-M3 | 2760 | 148 | 17/17 | 0.999990 | 0.0017 | 87 (0.018) |
| Ling-3.1-flash-base | 3952 | 152 | 384/384 | 0.983436 | 0.0004 | 192 (0.025) |

The RNG reconstruction is correct. That part is not in dispute.

What is in dispute is what it means — and his own README says it:

> "This is the GPU that created the starting weights. It doesn't say where the training
> happened after that."

That is the claim. The initialization configuration. Not the training cluster, not the
location, not the acquisition, not an export violation. The thread's 2,000–3,000 GPUs and
half a billion dollars are not in the repo.

## 1. The Kimi GPU check cannot reproduce the Kimi result

`replay.py` passes a per-model RNG offset:

```python
out.append(corr(y, draw(pos, sms * 2048, seed, m.get("cpu_offset", 0), m["dist"])))
```

`models.json` gives Kimi `"cpu_offset": 3776` and no other model one. Using his own `draw()`
and `corr()` on his own rows:

```
offset=    0  median r=-0.008076
offset= 3776  median r= 0.999996
```

But `gpu_check.py` does not read `cpu_offset` at all. It calls `torch.cuda.manual_seed(seed)`
and then `normal_()`, which starts at offset 0. The advertised **"GB200 13/13"** therefore
cannot be produced by the committed `gpu_check.py`; it needs an uncommitted modification, or
it was produced by a different script. The fix is small — consume the preceding RNG — but it
is not in the repo.

## 2. "Bit for bit" is not tested by any code in the repo

`gpu_check.py` compares with correlation and counts rows above 0.99:

```python
r, ctl = corr(gen(m, 0), Y), corr(gen(m, 1), Y)
print(f"... {(r > 0.99).sum()}/{len(r)} ... {np.median(r):.6f}")
```

There is no `torch.equal`, `array_equal`, or `allclose` anywhere. The README's "8/8, bit for
bit" is not produced by any code in it.

For DeepSeek-V4-Flash it is worse. `fetch.py` asserts BF16 storage and returns bf16-rounded
rows:

```python
assert info["dtype"] == "BF16"
return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
```

while `gpu_check.gen` creates the tensor in the model's declared dtype — `float32` for V4. A
bit-exact comparison between bf16-stored rows and float32-generated values is impossible on
that path. The 461/469 is a correlation threshold, not a bit match.

## 3. The search is not in the repo

There is no script for the claimed 2^32 seed brute force. `replay.py` reads the seeds from
`models.json` and verifies them. That is enough to check the result, but "brute-forced over all
2^32 values" is not reproducible from the repo. The Kimi offset is likewise undocumented: it
is the one extra free parameter, it is required for the match, and nothing says where it came
from.

## 4. The fingerprint is a software parameter

`replay.py` regenerates the "hardware fingerprint" on a laptop in seconds; the README
advertises exactly that ("runs on a laptop in a few seconds"). The matched rows are, by the
method's own construction, embedding rows for tokens that never appeared in training — the
parameters that never received a gradient. So the method recovers the initialization of unused
parameters, not the hardware that trained the model.

And the SM count is a class, not a part number. His notes:

> "H100, H800 and H200 are all 132 SMs. GB200 and GB300 are both 152 and so is an AMD MI300X
> in two-way partition mode."

## The sentence that matters

> "This is the GPU that created the starting weights. It doesn't say where the training
> happened after that."

A fingerprint that can be generated on a laptop, from initialization parameters that were
never trained, identifies a software configuration. It is consistent with a rented offshore
node, a partner lab, a deliberate decoy, or a smuggled cluster. It does not distinguish them.
That was the original objection, and the repo now states it in its own notes.

## Reproduce it

```bash
git clone https://github.com/macrogrants/checkpoint-forensics
cd checkpoint-forensics
python replay.py

python - <<'PY'
import json, numpy as np
from gpuid import draw, corr
from fetch import rows
m = [x for x in json.load(open("models.json")) if x["name"] == "Kimi-K3"][0]
Y = rows(m)
for off in (0, m["cpu_offset"]):
    rs = []
    for r, y in zip(m["rows"], Y):
        pos = np.uint64(r * m["shape"][1]) + np.arange(m["shape"][1], dtype=np.uint64)
        rs.append(corr(y, draw(pos, m["sms"] * 2048, m["seed"], off, m["dist"])))
    print(f"offset={off:5d}  median r={np.median(rs):.9f}")
PY
```

## What would change the conclusion

A committed script that reproduces the Kimi 13/13 on a 152-SM GPU with the required offset,
and a bit-equality test for the models claimed bit-for-bit. Until then the repo supports a
much smaller claim than the thread:

**these rows are consistent with `normal_(seed)` under an SM-count class at initialization.**

That is a real and clever result. It is not evidence of a smuggled Blackwell cluster.
