"""Run every experiment in order and regenerate the figure."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["exp1_validate.py", "exp2_forge.py", "exp3_train.py", "exp4_stats.py", "make_figures.py"]

if __name__ == "__main__":
    t_all = time.time()
    for s in STEPS:
        print(f"\n{'='*78}\n== {s}\n{'='*78}")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"{s} failed with code {r.returncode}")
        print(f"-- {s} done in {time.time()-t0:.1f}s")
    print(f"\nall experiments complete in {time.time()-t_all:.1f}s")
