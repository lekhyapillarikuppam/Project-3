# CLAUDE.md: instructions for Claude Code

## What this project is
A course project (Applied Data Science and AI, IISc). Read `SPEC.md` first. It is the source of truth.
Short version: evaluate zero-shot CLIP and OWLv2 on crime-scene photos (FFS dataset), using ground truth built from numbered markers and exhibit lists.

## How to work
- Spec first. Before writing code for a section, make sure `SPEC.md` describes it. If code and spec disagree, update the spec first, then the code, and add a line to the change log (section 10). Record choices in the decisions log (section 9).
- Work only on the phase the user names. Current status: Phase 0 (setup) is done. Do not start Phase 1 or later until the user says so.
- Keep code simple and readable. The user must be able to explain every line. Prefer short functions with clear names and comments that say why, not what.
- Add or extend a pytest test for each acceptance test in the spec.
- Heavy compute (OWLv2 over all photos) runs on Kaggle or Colab, not locally. Write code in `src/ffs/` so notebooks only import and call it.

## Hard rules
- Never edit anything under `data/raw/`.
- Do not run models on test scenes until the protocol is frozen (git tag `protocol-v1`). Tune prompts and thresholds on dev scenes only.
- Do not commit photos, raw spreadsheets, model weights, or large result files. See `.gitignore`.
- Stage B (detectors) must never read silver or gold labels, or exhibit lists.
- Call stains "red-brown stain", never "blood". Do not detect or describe mannequin bodies as people.
- Outputs are "candidate observations". Do not claim an object was used in a crime.

## Conventions
- Python 3.10+. Paths come from `src/ffs/config.py`. The random seed is `SEED` in that file.
- CSV schemas are in `SPEC.md` section 2 and in `src/ffs/schemas.py`. Do not change a schema without updating the spec and the code together.
- Run `pytest` before saying a task is done.
- Laptop is Windows with no NVIDIA GPU. Use PowerShell-friendly commands and `pathlib` for paths.
