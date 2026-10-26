# Phase 3 steps (Windows, PowerShell): gold labels and Stage B on dev

Run from the repo folder with the virtual environment active.

## A. Hand-label the test photos (your job, about 1.5 to 2 hours, in sessions)

1. `python scripts/make_gold_tool.py`
2. Double-click `results\gold\gold_tool_pass1.html` (opens in Chrome). Keep it in that folder: it reads the
   photos from `..\..\data\raw\`.
3. For each photo press the number keys for every category that is present (1 to 5), then Enter. Click or
   press Z to zoom. Press L only if you are unsure. Silver labels are hidden on purpose.
4. Labels are kept in the browser as you go. Press S at the end of each session: it downloads
   `gold_pass1_export.csv` (it also downloads by itself every 100 photos). Move it into `results\gold\`.
   Do not clear Chrome's site data until the labels are imported.
5. Progress check: `python scripts/import_gold.py results\gold\gold_pass1_export.csv --allow-partial --dry-run`
6. When all 545 are done: `python scripts/import_gold.py results\gold\gold_pass1_export.csv` writes `data_meta\gold.csv`.
7. At least one day later: `python scripts/make_gold_tool.py --pass2 55`, label that page the same way, save its
   export, then `python scripts/import_gold.py --agreement results\gold\gold_pass2_export.csv`.

## B. Stage B on dev (Kaggle, about 30 minutes)

1. `python scripts/package_stage_b.py --split dev` makes `results\stage_b\ffs-stage-b-dev.zip`.
2. kaggle.com: sign in (phone-verified account needed for Internet and GPU). Create, New Dataset, upload the zip,
   name it `ffs-stage-b-dev`, visibility **Private** (the photos are CC BY-NC-SA, non-commercial).
3. Create, New Notebook. Add Input: your dataset. Settings: Accelerator GPU, Internet On.
4. Cell 1: `!find /kaggle/input -maxdepth 3 -name manifest.csv` (shows the dataset folder; use it below).
5. Cell 2 (smoke test, 5 photos): `!python <folder>/stage_b_run.py --root <folder> --name dev --out /kaggle/working --limit 5`
   If it errors, paste the error in the chat. This code has not run on a real GPU yet.
6. Cell 3 (full): the same command without `--limit`.
7. Output tab: download `clip_scores_dev.csv`, `owlv2_raw_dev.csv`, `run_info_dev.json` into one folder, then
   `python scripts/import_stage_b.py <that folder> --name dev`.
