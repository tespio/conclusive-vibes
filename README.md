# The Blackwell "weight fingerprint", audited

A runnable proof that the *checkpoint forensics* claim — "Chinese labs initialised
their models on export-controlled NVIDIA Blackwell GPUs, as proven by the weights" —
identifies a **software launch configuration**, not hardware.

**Headline result:** a checkpoint generated on a laptop CPU, with no GPU in the
loop, is certified by the claimed method as **GB200, 256/256 values bit-identical,
"conclusive"**. A single optimizer step then destroys that fingerprint for 88% of
parameters, and by step 2 the method cannot recover the seed at all.

- Read the full write-up: **[REPORT.md](REPORT.md)**
- Figure: [figures/fig1_audit.png](figures/fig1_audit.png)
- Raw numbers: `results/*.json`

## Interactive site

`site/` is a three.js scrollytelling version of the audit — the integer fingerprint, the
CPU forgery, the one-step collapse, and the σ arithmetic, all playable in the browser.
Selecting a GPU in the first exhibit re-lays the 3D wall as `SMs × blocks/SM` cells and
updates the grid readout and barcode with it. The footer links open the full report, all
13 Python sources under `poc/`, and the audit figure in in-page viewers: a markdown reader,
a file browser with highlighted source (copy plus per-file download), and a zoomable figure.
A floating button returns to the top.

```bash
python -m http.server 8123        # from the repo root
# open http://127.0.0.1:8123/site/
```

![Site preview](site/preview.png)

The figures on the site are the measured values from `results/*.json`. three.js is vendored
locally, so the page has no external dependencies.

**Publish:** the repo is GitHub Pages–ready — `build_site.py` assembles the artifact and
`.github/workflows/pages.yml` deploys it. See [PUBLISHING.md](PUBLISHING.md) for the one-time
setup and why it must deploy via Actions rather than *Deploy from a branch*.

## Quick start

```bash
pip install -r requirements.txt
cd poc
python run_all.py        # ~150 s on an RTX 4070 Ti SUPER; regenerate everything
```

The only hardware requirement is any CUDA GPU for the validation and training
experiments. `exp2` (the forgery itself) runs on CPU alone.

## What each piece does

| file | role |
|---|---|
| `poc/philox.py` | bit-exact host reproduction of PyTorch's CUDA Philox stream and launch policy |
| `poc/model.py` | 10.2M-parameter GPT whose init is reproducible on GPU **or** CPU |
| `poc/detector.py` | the claimed forensic method, reimplemented faithfully |
| `poc/exp1_validate.py` | proves the reproduction is exact (10,244,352/10,244,352 params) |
| `poc/exp2_forge.py` | forges GB200 / H100 / B300 checkpoints on CPU; runs the detector |
| `poc/exp3_train.py` | measures fingerprint decay under real training |
| `poc/exp4_stats.py` | audits the σ / correlation / sample-size arithmetic |
| `poc/probes/` | the reverse-engineering probes used to find curand's counter layout and FMA behaviour |

## The short argument

1. PyTorch sizes its RNG launch grid as `min(ceil(N/256), SMs × (threads_per_SM/256))`.
   That integer is the **only** way hardware enters the initialization stream.
2. We reproduce a real CUDA initialization **bit-for-bit on CPU**, so the "hardware
   verification" step proves nothing that a laptop cannot also produce.
3. The recovered object is the stream that *initialised* the model. Training erases
   it from every parameter that receives a gradient; it survives only in parameters
   that were never trained — exactly what a forger leaves behind.
4. The reported σ values are mathematically inconsistent with the reported
   correlation, and are dominated by sample count rather than hardware.
