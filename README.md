# AI-Assisted Evidence Inventory with Marker-Based Ground Truth

Course project, Applied Data Science and AI (IISc).

**Question:** how well does a zero-shot AI (CLIP, OWLv2) find evidence categories in photos of staged
crime scenes, when the answer key is built from the scenes' own numbered markers and exhibit lists, and
how trustworthy is that answer key?

All outputs are *candidate observations* for a human to verify. Nothing here says an object was used in a crime.

## Start here

1. `docs/PHASE0_SETUP.md` for laptop setup (Windows).
2. `SPEC.md` is the source of truth (spec-driven development).
3. `TASKS.md` is the phase checklist.

## Phases

| Phase | What | Status |
| --- | --- | --- |
| 0 | Foundation and spec skeleton | done when the gate in `docs/PHASE0_SETUP.md` passes |
| 1 | Feasibility gate | not started |
| 2 | Data and categories | not started |
| 3 | Stage A: silver labels | not started |
| 4 | Gold labels and detectors on dev | not started |
| 5 | Test evaluation and analysis | not started |
| 6 | Web app, deploy, wrap-up | not started |

## Layout

```
SPEC.md            the growing spec
CLAUDE.md          instructions for Claude Code
src/ffs/           code (config, schemas, later stage_a, stage_b, eval)
tests/             pytest tests, one per spec acceptance test
scripts/           small command-line scripts (check_env.py)
data/              raw photos and downloads (not committed)
data_meta/         small CSVs: exhibits, splits, markers, labels
results/           metrics, figures, raw detections
notebooks/         only for Kaggle/Colab GPU steps
app/               static web app (Phase 6)
docs/              setup and phase notes
```

## Dataset and license

Fictional Forensic Scenes (FFS), School of Criminal Justice, University of Lausanne.
Zenodo record: https://zenodo.org/records/17151982
Dataset paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC13200034/
License: CC BY-NC-SA 4.0. Credit the dataset; keep the same license on published labels; non-commercial use only.

Content note: some mannequins resemble real bodies, and scenes contain simulated blood.
