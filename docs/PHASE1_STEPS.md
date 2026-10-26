# Phase 1: feasibility gate (Windows, PowerShell)

Goal: find out, on a few real scenes, whether the colour + shape marker finder and the
constrained OCR read are good enough to build silver labels. Everything runs on your laptop.
Time: about 2-3 hours, most of it the hand-labelling in step 8.

Run commands from the repo folder with the virtual environment active (prompt starts with `(.venv)`).
If not: `.venv\Scripts\Activate.ps1`

## What was built (read `SPEC.md` sections 4 and 8 first)

| Piece | Where |
| --- | --- |
| Pull only photos + English lists out of the 45 GB remote zip | `src/ffs/remote_zip.py`, `scripts/fetch_scenes.py` |
| Read an exhibit-list spreadsheet | `src/ffs/exhibits.py`, `scripts/inspect_exhibits.py` |
| Find yellow numbered markers | `src/ffs/markers.py` |
| Read the number, check it against the list | `src/ffs/ocr.py` |
| Run it all on scenes and write overlays | `scripts/run_marker_probe.py` |
| Compute pass/fail | `src/ffs/feasibility.py`, `scripts/feasibility_report.py` |

Honest limits: I could not reach Zenodo from my workspace, so the download code and the real
EasyOCR reading were **not** tested on real data. The marker finder, the constrained read, the
spreadsheet reader and the metrics are tested on synthetic data (44 tests). Step 3 is a checkpoint
for exactly this reason.

## Step 1. Add the Phase 1 files

1. Download `ffs-phase1-update.zip`.
2. In PowerShell, from the repo folder (change the Downloads path if needed):

```
Expand-Archive -Path "$HOME\Downloads\ffs-phase1-update.zip" -DestinationPath . -Force
git status
git add -A
git commit -m "Phase 1: marker finder, OCR, remote zip fetch, feasibility scripts"
git push
```

`git status` should list new and changed files only (SPEC.md, requirements.txt, src/, scripts/, tests/, docs/).

## Step 2. Install the ML packages and run the tests

```
pip install -r requirements.txt
pip install -r requirements-ml.txt
pytest
```

- The ML install is large (a few hundred MB to 2 GB) and takes several minutes.
- Expect `44 passed`. (Before the ML install one test is skipped; after it, it runs.)
- The first OCR run downloads EasyOCR model files (about 100 MB). That needs internet.

If the OCR test fails, copy the failure and paste it in the chat.

## Step 3. Test the pipeline on a fake scene (no download)

```
python scripts/make_demo_scene.py
python scripts/run_marker_probe.py --scenes th_9999
```

Open `results\feasibility\overlays\th_9999\` and look at the images. Each box shows
`final number (raw OCR read)`. The photos contain 7; 3 and 12; 18; nothing; 9.
Expected: 7, 3, 12 and 9 are read correctly. The "18" is not in the demo exhibit list, so it
should snap to 16 or be rejected. That is the constrained read working, not a bug.

If boxes are missing or the reads are wrong, tell me before going on.

Then delete the demo: `Remove-Item -Recurse data\raw\th_9999, results\feasibility`

## Step 4. CHECKPOINT: look at the real zip layout

```
python scripts/fetch_scenes.py list
```

This reads only the zip's table of contents (a few MB). It prints the folder layout, counts per
kind (photo, exhibit list, blocked) and photos per scene, and saves it to `data\raw\`.

**Paste the whole output into the chat and stop.** I have not seen the real layout, so I will
confirm it and update `SPEC.md` section 4.1 (and the code, if needed) before you download anything big.

## Step 5. Choose 5 scenes

(After the checkpoint.) Download the description sheets and one scene:

```
python scripts/fetch_scenes.py pull --scenes th_0001 --dry-run
python scripts/fetch_scenes.py pull --scenes th_0001
```

Open the description spreadsheet in `data\raw\` in Excel. Pick **4 scenes with markers and photos**
(prefer scenes with 3 or more markers) and **1 scene with no markers**. Tell me the five ids.

## Step 6. Download them and check the exhibit lists

```
python scripts/fetch_scenes.py pull --scenes th_0003 th_0007 th_0011 th_0014 th_0018 --dry-run
python scripts/fetch_scenes.py pull --scenes th_0003 th_0007 th_0011 th_0014 th_0018
python scripts/inspect_exhibits.py --scene th_0003
```

(Use your own five ids.) `inspect_exhibits.py` prints the spreadsheet and which columns it chose as
marker number and description. Check by eye that they are right. If not, rerun with
`--marker-col N --desc-col M` and tell me.

## Step 7. Run the probe on the real scenes

```
python scripts/run_marker_probe.py --scenes th_0003 th_0007 th_0011 th_0014 th_0018
```

Quick finder-only test first, if you want: add `--no-ocr`. The full run can take 10-30 minutes on the CPU.

It writes `results\feasibility\overlays\` (photos with boxes drawn), `probe_markers.csv`,
`probe_photos.csv` and a blank `truth.csv`.

## Step 8. Fill in the truth sheet (the honest check)

Open `results\feasibility\truth.csv` in Excel. It lists about 40 photos. For each row:

1. Open the **original** photo (the `relpath` column, under `data\raw\`). Do **not** start from the overlay,
   so the machine's answer does not bias you.
2. `n_markers_visible`: how many numbered yellow markers you can clearly see.
3. `true_marker_nos`: their numbers, separated by `;` (for example `3;12`). Leave blank if none.
4. `notes`: optional (blurry, marker cut off, and so on).

Save as CSV (keep the same file name and columns).

Then open the overlays to see where the finder and OCR went wrong.

## Step 9. Get the verdict

```
python scripts/feasibility_report.py
```

It prints the finder hit rate and the constrained read accuracy against the pass marks
(0.70 and 0.60) and says what to do next (`SPEC.md` section 8.3). Paste the output in the chat.

If it fails, do not tune anything yet. Send me the report and a few overlays, and we will tune together,
logging each change in the decisions log.

Commit your results (the overlays are ignored by git on purpose):

```
git add -A
git commit -m "Phase 1: feasibility results"
git push
```

## Phase 1 gate

- [ ] `pytest` passes (44 passed)
- [ ] Demo scene reads look right
- [ ] Real zip layout pasted in chat and confirmed
- [ ] 5 scenes downloaded; exhibit columns confirmed
- [ ] `truth.csv` filled for about 40 photos
- [ ] `feasibility_report.py` run and pasted in chat
- [ ] Decision made (pass, tune, or fallback) and logged in `SPEC.md` section 9
