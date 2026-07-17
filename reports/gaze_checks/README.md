# Gaze visual checks

Generated HTML overlays live here (`gate1/`, `gate2/`).

```bash
conda activate gnn-gaze
python scripts/gaze_overlay_check.py --gate 1   # metadata–gaze alignment
python scripts/gaze_overlay_check.py --gate 2   # fixation→segment assignment
```

Open `gate1/index.html` or `gate2/index.html`. Owner sign-offs go in `reports/DECISIONS.md`.
