"""Assemble the static site artifact that GitHub Pages publishes.

Layout is preserved so the page's relative paths keep working:

    _site/index.html            -> redirects to ./site/
    _site/site/...              -> the three.js site
    _site/REPORT.md             -> fetched by the report viewer
    _site/poc/**.py             -> fetched by the code browser
    _site/figures/fig1_audit.png
    _site/results/*.json        -> linked from the report

The 39 MB forged checkpoint (`results/forged_gb200_step0.npz`) is deliberately
left out: it is a generated artifact, reproducible with `poc/exp2_forge.py`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_site"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    copy_file(ROOT / "index.html", OUT / "index.html")
    copy_file(ROOT / "REPORT.md", OUT / "REPORT.md")
    copy_file(ROOT / "REBUTTAL.md", OUT / "REBUTTAL.md")
    shutil.copytree(ROOT / "site", OUT / "site")
    copy_file(ROOT / "figures" / "fig1_audit.png", OUT / "figures" / "fig1_audit.png")

    for p in sorted((ROOT / "results").glob("*.json")):
        copy_file(p, OUT / "results" / p.name)

    for p in sorted((ROOT / "poc").rglob("*.py")):
        copy_file(p, OUT / "poc" / p.relative_to(ROOT / "poc"))

    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"built {OUT} — {len(files)} files, {total / 1024 / 1024:.2f} MB")
    for p in files:
        print(" ", p.relative_to(OUT).as_posix())


if __name__ == "__main__":
    main()
