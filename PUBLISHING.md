# Publishing to GitHub Pages

The site is static and has no build step, but it must be published as a **prebuilt
artifact** rather than with *Deploy from a branch*.

**Why:** branch deploys run Jekyll, which converts `REPORT.md` into `REPORT.html`.
The in-page report viewer fetches `../REPORT.md`, and the code browser fetches
`../poc/*.py`; both would break. The included workflow uploads `_site/` directly,
so GitHub Pages serves the files untouched.

## One-time setup

1. Create an empty repository on GitHub (no README, no `.gitignore`).
2. From this folder:

   ```bash
   git init -b main
   git add .
   git commit -m "Checkpoint forensics audit: PoC + interactive site"
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```

3. In the repository: **Settings → Pages → Build and deployment → Source =
   GitHub Actions**. Do not pick “Deploy from a branch”.
4. The push in step 2 already triggered the workflow; if you changed the setting
   afterwards, run **Actions → Deploy site → Run workflow**.

The landing URL is `https://<user>.github.io/<repo>/`, which redirects to
`/site/`. Custom domains work the same way — no code changes needed, because every
path in the site is relative.

## What gets published

`build_site.py` assembles `_site/` with the exact layout the page expects:

```
_site/index.html                  redirect to ./site/
_site/site/                       the three.js site (three.js and marked vendored)
_site/REPORT.md                   fetched by the report viewer
_site/poc/**/*.py                 fetched by the 13-file code browser
_site/figures/fig1_audit.png
_site/results/*.json              linked from the report
```

Total ≈ 2.7 MB. `tools/`, `_site/`, `__pycache__/` and the 39 MB
`results/forged_gb200_step0.npz` are excluded; the checkpoint is reproducible with
`python poc/exp2_forge.py`.

## Preview the exact artifact locally

```bash
python build_site.py
python -m http.server 8124 --directory _site
# open http://127.0.0.1:8124/
```

Serving from `_site/` rather than the repository root is the real test: it is what
a Pages deploy looks like. If the report or a source file fails to load there, the
artifact is missing something.

## Updating

Push to `main`; the workflow rebuilds and redeploys. The workflow can also be run
manually from the Actions tab.
