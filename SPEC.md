# SPEC: AI-Assisted Evidence Inventory with Marker-Based Ground Truth

Status: Phase 1. Sections 1-3, 4.1-4.4 and 8 are written; 4.1 is confirmed against the real zip. Other sections are stubs that
get filled in the phase named next to each one, **before** the code for that phase is written.

Rule: if code and spec disagree, change the spec first, then the code, and add a
line to the Change log (section 10).

---

## 1. Purpose and non-goals

### 1.1 Purpose

Measure how well an off-the-shelf (zero-shot) AI finds evidence categories in
photographs of staged crime scenes, when the answer key is built from the scene
itself: numbered yellow markers placed next to each item, looked up in the
investigators' exhibit list.

Research question:

> How accurately can CLIP and OWLv2 (no training) tell which evidence categories
> appear in a photo of a staged scene, and how trustworthy is the marker-based
> answer key they are scored against?

### 1.2 Dataset

Fictional Forensic Scenes (FFS), University of Lausanne, on Zenodo.
License CC BY-NC-SA 4.0: credit the dataset, keep the same license on any
published labels, non-commercial use only.

Parts used: documentation photos (`img/`), exhibit lists (`IPC/IPC_ENG/`),
description sheets (`description/`, scene selection only).
Parts not used: panoramas and laser scans.

### 1.3 Goals

- G1. Build photo-level labels (silver) from markers plus exhibit lists, and
  measure their accuracy against hand-corrected labels (gold).
- G2. Score CLIP (baseline) and OWLv2 (main) per category against gold.
- G3. Report results by photo type (overview, mid-range, close-up) and under a
  marker-masking experiment with a random-patch control.
- G4. Analyse failures honestly and present results in a small static web app.

### 1.4 Non-goals

- N1. Deciding whether any object was used in a crime.
- N2. Reconstructing events, predicting crime type, identifying anyone.
- N3. Counting objects across photos (presence only, never counts).
- N4. Training or fine-tuning any model.
- N5. Using the 3D scans or panoramas.
- N6. Scoring box quality (FFS has no box annotations).

### 1.5 Safety statement (shown in the app)

All outputs are "candidate observations" for a human to verify. Stains are
called "red-brown stain", never "blood". Some mannequins resemble real bodies;
the dataset contains simulated blood.

---

## 2. Data contracts

All files are UTF-8 CSV unless noted. Column names are exact. The Python
definitions live in `src/ffs/schemas.py` and are checked by `tests/test_schemas.py`.
Allowed values are in `src/ffs/config.py`.

Identifiers: `scene_id` is the dataset's scene folder name (e.g. `th_0024`).
`photo_id` is the photo file name without extension (e.g. `IMG_0031`).

| File | One row per | Columns |
| --- | --- | --- |
| `data_meta/exhibits.csv` | exhibit-list row | `scene_id, marker_no, description` |
| `data_meta/category_rules.csv` | mapping rule | `pattern, category, version` |
| `data_meta/exhibit_categories.csv` | exhibit | `scene_id, marker_no, category, source, confirmed` |
| `data_meta/scene_split.csv` | scene | `scene_id, split, has_markers, n_photos` |
| `data_meta/photos.csv` | photo | `scene_id, photo_id, relpath, width, height, photo_type` |
| `data_meta/markers.csv` | marker found in a photo | `scene_id, photo_id, marker_idx, x, y, w, h, method, det_score, raw_ocr, ocr_conf, marker_no, status` |
| `data_meta/silver.csv` | photo x category | `scene_id, photo_id, category, present, marker_nos` |
| `data_meta/gold.csv` | photo x category (test scenes only) | `scene_id, photo_id, category, present, corrected, labelled_at` |
| `results/clip_scores.csv` | photo x category x prompt | `scene_id, photo_id, category, prompt, score` |
| `results/owlv2_raw.csv` | raw box | `scene_id, photo_id, category, prompt, score, x, y, w, h, tile` |

Value rules:

- `split` is `dev` or `test`. Split is by scene, never by photo.
- `photo_type` is `overview`, `mid`, `close` or `unknown`.
- `present` and `corrected` and `confirmed` and `has_markers` are 0 or 1.
- `source` (in `exhibit_categories.csv`) is `rule`, `llm` or `manual`.
- `status` (in `markers.csv`) is `accepted`, `snapped` or `rejected`.
- `method` (in `markers.csv`) is `color_shape` or `owlv2`.
- `category` is one of the final categories (section 3) or `other`.
- `silver.csv` and `gold.csv` are full matrices: every photo has one row for
  every final category (not `other`), so absent categories appear with `present=0`.
- `marker_nos` is a `;`-separated list of marker numbers (may be empty).
- Box coordinates `x, y, w, h` are pixels in the original photo.

Design rule: OWLv2 is run once on a GPU with a low score cut-off and **all**
boxes are saved to `owlv2_raw.csv`. Thresholds are tuned afterwards on a laptop
from that file, so the model never has to be re-run.

---

## 3. Category definitions

### 3.1 Candidate categories

Final 6-8 are chosen in Phase 2 after counting the exhibit lists.

| Category (`category` value) | Typical exhibit wording |
| --- | --- |
| `blade` | knife, cutter, scissors |
| `firearm` | pistol, rifle |
| `ammunition` | cartridge, casing (may be too small; check) |
| `bottle_glass` | bottle of vodka, glass |
| `phone_electronics` | mobile phone, laptop |
| `paper_document` | A4 sheets, handwritten note |
| `footwear` | shoe, boot |
| `red_brown_stain` | reddish trace, stain |

### 3.2 Inclusion rule

A category is kept only if the **test** split has at least **15 positive photos
from at least 6 scenes** (decision D-003). Otherwise it is merged into `other`.

### 3.3 Other

Exhibits that fit no kept category map to `other`. They count toward scene
coverage but are excluded from per-category metrics.

### 3.4 Mannequin bodies

Not detected. Body exhibits stay in exhibit counts and map to `other`.

### 3.5 Mapping rules

A written rule table (`category_rules.csv`, regex or keyword patterns,
versioned) maps exhibit descriptions to categories. Descriptions that no rule
covers are suggested by an LLM, confirmed by hand, and recorded with
`source=llm` and `confirmed=1`. Every label is traceable to a rule, an LLM
suggestion or a manual decision.

---

## 4. Stage A: marker reader and answer key  (4.1-4.4 written in Phase 1; 4.5 in Phase 3)

### 4.1 Raw data layout (confirmed 2026-10-26 from `fetch_scenes.py list`, part 1)

The Zenodo part-1 zip (`th_0001-0020.zip`, scenes th_0001 to th_0020) has one
top-level folder, `th_0001-0020/`, containing:

```
th_0001-0020/
  Read_me_ENG.pdf, Read_me_FR.pdf
  description/Forensic_scene_th_0001-0020_ENG.xlsx   one sheet for all 20 scenes (+ _FR)
  IPC/IPC_ENG/IPC_Th_0001.xlsx ... IPC_Th_0020.xlsx   one exhibit list per scene (+ IPC_FR/)
  img/th_0002/IMG_20240227_085541.jpg ...             photos, one folder per scene
  (panoramas and scans: about 41 GB, never downloaded)
```

Facts that follow from the listing:

- Scene th_0001 has an exhibit list but **no photos** (consistent with the dataset
  paper: some scenes have no documentation photos). Scenes th_0002 to th_0020 have photos.
- The archive contains many macOS junk files (`._*`, `.DS_Store`). They are ignored.
- Photo names are camera timestamps (`IMG_YYYYMMDD_HHMMSS.jpg`), so name order is
  shooting order. Phase 2 uses this to tag photo types by the photo protocol
  (overview shots first, then close-ups).
- Each exhibit list is a separate `.xlsx`; the scene id is in the file name (`IPC_Th_0002`).
- From the description sheet (read in Phase 1): every scene th_0002 to th_0020 has between 3 and 14
  markers, all interiors. **Part 1 has no marker-free scene.** Marker-free photos exist inside scenes;
  marker-free scenes (for false-positive measurement) must be looked for in parts 2-4 in Phase 2.
- Marker numbers are mostly **two-digit** (11-21, 30-35, 60, 69, 72, 90) and the same numbers recur
  across scenes. A few lists repeat a number (th_0014: 15 twice, th_0015: 13 twice). Keep every
  exhibit row; a repeated number maps one marker to several exhibits.

How code recognises files (case-insensitive, by path component names):

- A scene is a folder named `th_NNNN` (photos), or `th_NNNN` / `Th_NNNN` as a
  separate token in a file name (exhibit lists). The archive folder `th_0001-0020`
  is **not** a scene (this was a bug found by the first real listing; D-009).
- Photos are `.jpg`/`.jpeg`/`.png` files under a folder named `img`.
- English exhibit lists are spreadsheets under `IPC_ENG`.
- Description sheets are English spreadsheets under `description`.
- Everything else is never downloaded: `pano`, `scan`, `IPC_FR`, French files, and junk.

Requirements:

- R-A0-1. Fetching never downloads `pano/` or `scan/` content.
- R-A0-2. Files are written only inside `data/raw/` (no path traversal).
- R-A0-3. Files already present with the same size are skipped.
- R-A0-4. macOS junk files (`._*`, `.DS_Store`) are never downloaded or read.
- R-A0-5. A file's scene is never taken from the archive's top-level folder name.

### 4.2 Exhibit lists (reading)

- R-A1-1. `read_exhibit_file(path)` returns columns `marker_no` (int) and
  `description` (str) from `.xlsx`, `.xls` or `.csv`.
- R-A1-2. The marker-number column is the one where most non-empty cells parse
  as whole numbers. The description column is the text column with the longest
  average text. The caller can override both by name.
- R-A1-3. Rows without a parsable number are skipped. One marker can cover a
  group ("4 white cans"); the quantity stays inside `description`.
- R-A1-4. Because the real column layout is not yet known, `scripts/inspect_exhibits.py`
  prints each file as the reader sees it, so the layout can be confirmed by eye.

### 4.3 Marker finder (A2, colour + shape)

Input: a BGR image. Output: boxes `x, y, w, h` in **original** pixels, each with a score.

- R-A2-1. Work on a downscaled copy (longest side at most `work_side`, default
  1600 px); map boxes back to original coordinates.
- R-A2-2. Yellow mask in HSV (OpenCV scale): H 16-40, S at least 150, V at least 55,
  then a small morphological close. (Values from D-013; they were H 18-38, S 110, V 110.)
- R-A2-3. A candidate blob is kept only if: area is between 0.00003 and 0.15 of
  the image; solidity (area / convex hull area) at least 0.60; bounding box
  width/height between 0.4 and 2.5. A triangle is **not** required, because a
  tent-shaped placard can look like a triangle, trapezoid or rectangle
  depending on the view (decision D-007).
- R-A2-4. Ink test: dark pixels (defined in R-A2-8) inside the blob outline must
  be between 2% and 45% of the blob area. This is the printed number; a plain
  yellow object without it is rejected.
- R-A2-5. Boxes overlapping with IoU above 0.30 are merged, keeping the higher score.
- R-A2-6. `score = solidity * (0.5 + 0.5 * min(1, ink_fraction / 0.08))`.
- R-A2-7. All thresholds live in `MarkerParams` (`src/ffs/markers.py`). Any change
  made after looking at real photos is logged in section 9.
- R-A2-8. "Dark" is relative to the marker's own brightness: a pixel inside the blob is ink if
  its V is below 0.5 times the median V of the blob's yellow body (the yellow-mask pixels of the
  blob), **and** below 90. The fixed rule "V below 90" alone failed on underexposed photos where the
  body itself is near V 100; the relative part fixes that (D-013). The cap of 90 keeps the old
  behaviour on well-exposed photos, where half the body V (about 127) would also count mid-grey
  background as ink and break the OCR preprocessing. The ink fraction limits of R-A2-4 are
  unchanged. The same rule is used to pick the digit pixels for OCR (R-A3-2).

Acceptance tests (synthetic images, `tests/test_markers.py`):

- AT-A2-1. A yellow marker with a black "7" on a cluttered background gives one box with IoU at least 0.5 against the truth.
- AT-A2-2. A plain yellow rectangle with no digits gives no box.
- AT-A2-3. An image with no yellow gives no box.
- AT-A2-4. For a large image the returned box is in original coordinates, not working-scale coordinates.
- AT-A2-5. Dark photo: a marker whose body has V about 100 (H 19, S 250) and digits with V about 20, on a
  dark background, gives one box with IoU at least 0.5.
- AT-A2-6. A marker whose outline is not compact (a yellow tail attached, as for a tilted or partly covered
  marker; solidity about 0.65) gives one box.

### 4.4 Reading the number (A3)

- R-A3-1. Crop = box grown by 10% on each side, clipped to the image.
- R-A3-2. Preprocessing: keep the dark ink pixels (dark as in R-A2-8) inside the yellow region,
  render them as black digits on white, scale so the height is at least 96 px,
  add a white border. Two variants are made: **tight** (the yellow region as found) and
  **wide** (the region grown by 12% of the crop size, so digits touching the marker's edge
  are not lost). Then crop to the bounding box of all ink plus a margin of 30% of the ink
  height, so the digits fill the image whatever the marker size (decision D-012), and scale so
  the ink is at least 80 px tall.
- R-A3-3. OCR is EasyOCR restricted to the digits `0123456789`. Pieces are joined
  left to right. `ocr_conf` is the mean piece confidence. The joined text is kept as `raw_ocr`.
- R-A3-4. Constrained read, given `valid` = the marker numbers of that scene's exhibit list:
  1. `raw_ocr` is in `valid`: status `accepted`, `marker_no = raw_ocr`.
  2. Otherwise take the valid numbers at edit distance 1 from `raw_ocr`
     (one digit changed, added or removed). If exactly one exists and
     `ocr_conf >= 0.40`: status `snapped`, `marker_no` = that number.
  3. If several exist, prefer the one whose changed digit is a visually confusable
     pair (0-8, 1-7, 3-8, 5-6, 6-8, 4-9). If still not unique: `rejected`.
  4. Anything else, including empty text: status `rejected`, `marker_no` empty.
- R-A3-5. A **one-digit** read is accepted only by exact match (rule 1). It is never
  snapped, because a single digit is within one edit of nearly every number on the
  list (decision D-010).
- R-A3-6. OCR runs on both preprocessing variants; the result with the best status
  (accepted, then snapped, then rejected) wins, ties broken by confidence (decision D-011).

Acceptance tests (`tests/test_ocr.py`):

- AT-A3-1. raw "7", valid {1, 7, 9}: accepted 7.
- AT-A3-2. raw "18", conf 0.8, valid {16, 7}: snapped 16.
- AT-A3-3. raw "18", conf 0.2, valid {16, 7}: rejected.
- AT-A3-4. raw "18", conf 0.8, valid {16, 19}: snapped 16 (8-6 is confusable, 8-9 is not).
- AT-A3-5. raw "", valid {1}: rejected.
- AT-A3-6. raw "123", conf 0.8, valid {12, 7}: snapped 12.
- AT-A3-7. raw "1", conf 1.0, valid {3, 7, 9, 12, 16}: rejected (R-A3-5; found in the Phase 1 demo).
- AT-A3-8. raw "1", valid {1, 7}: accepted 1.
- AT-A3-9. Two reads ("1" conf 1.0 and "12" conf 0.8), valid {12, 16}: the "12" read wins (R-A3-6).
- AT-A3-11. After preprocessing, the ink bounding box is at least 40% of the image height
  (before the white border), for a small digit on a large marker.
- AT-A3-10. Synthetic two-digit markers keep both digits after preprocessing (two ink components).

### 4.5 Silver and gold labels  (Phase 3 and 4)

TBD. Silver label construction from markers and exhibit categories, gold
correction procedure, and the acceptance test: given a photo where marker 7 is
read and exhibit 7 is a knife, silver has `blade` present for that photo.

## 5. Stage B: detectors  (filled in Phase 4)

TBD. Models (CLIP, OWLv2), prompts per category, tiling, thresholds
(confident / uncertain / dropped), box merge rule.

## 6. Evaluation protocol  (filled in Phase 4, frozen by git tag `protocol-v1`)

TBD. Planned content (from the project plan): scene-level dev/test split
(about 40/60, fixed seed); per-category precision, recall, F1 and AP; macro
average; scene-level bootstrap with 1000 resamples for 95% intervals and a
paired CLIP vs OWLv2 comparison; always-present baseline; silver vs gold
agreement (precision, recall, Cohen's kappa); scores against silver and gold;
slices by photo type and marker read; marker-masking experiment with a
random-patch control; false-positive rate on marker-free scenes; error analysis
of about 30 false positives and 30 false negatives per method.

## 7. Web app  (filled in Phase 6)

TBD. Static site over precomputed results. Must have: landing, scene viewer,
results, about and limits. Stretch: verification, scene summary, export,
captions, exhibit-list cross-check.

## 8. Feasibility gate  (Phase 1)

Question: can colour + shape marker finding and the constrained OCR read recover
enough marker numbers to build silver labels?

### 8.1 Procedure

1. `python scripts/fetch_scenes.py list` prints the layout of the remote zip. Confirm section 4.1.
2. `python scripts/fetch_scenes.py pull --scenes ...` downloads photos and English
   exhibit lists for 3-5 scenes (part 1 has no marker-free scene; the truth sample includes marker-free photos).
3. `python scripts/run_marker_probe.py --scenes ...` finds markers, reads numbers,
   writes `results/feasibility/probe_markers.csv`, `probe_photos.csv`, annotated
   overlay images, and a blank `truth.csv` (never overwritten, so your labels are safe).
4. Hand-fill the truth file for a sample of about 40 photos (at least 8 per
   scene; mostly close-ups with a marker in frame, plus about 5 with no marker):
   `n_markers_visible` and `true_marker_nos` (`;`-separated). Label from the
   original photo, not the overlay, so the machine's answer does not bias you.
5. `python scripts/feasibility_report.py` prints the metrics and the pass/fail verdict.

The feasibility sample does not involve Stage B, so tuning marker parameters on
these photos does not leak into the detector evaluation.

### 8.2 Metrics (over the sampled photos)

- **Finder hit rate** H = sum over photos of min(boxes found, n_markers_visible) / sum of n_markers_visible.
- **Finder false positives per photo** = sum of max(0, boxes found - n_markers_visible) / number of photos.
- **Constrained read accuracy** A = (true numbers present in the photo's final read numbers) / (all true numbers).
- **Raw read accuracy** = the same, but counting only exact `raw_ocr` matches, to show what the constrained read adds.

Pass marks (proposed): H at least 0.70 **and** A at least 0.60.

AT-F-1 (`tests/test_feasibility.py`): the metrics match a hand-computed example.
A synthetic demo scene (`scripts/make_demo_scene.py`) lets the whole flow be
tried without any download.

### 8.3 Decision table

| Outcome | Action |
| --- | --- |
| H and A both pass | Continue to Phase 2 with colour + shape; log the final parameters |
| H fails | Tune `MarkerParams` using the overlays (at most 2 rounds, log each change). If still failing, use OWLv2 to find markers (Kaggle notebook, added only if needed) |
| A fails, H passes | Improve preprocessing (rotation, contrast) for at most 2 rounds, then try PaddleOCR |
| Still failing | Fall back to scene-level ground truth (a scene is positive for a category if any photo shows it, checked by hand on test scenes) |

---

## 9. Decisions log

| ID | Date | Decision | Why |
| --- | --- | --- | --- |
| D-001 | 2026-10-26 | Work in a git repo on the laptop; Kaggle/Colab only when necessary | Laptop has no NVIDIA GPU; most steps are light |
| D-002 | 2026-10-26 | Run OWLv2 once on Kaggle, cache all raw boxes, tune thresholds locally | Avoids re-running the slow model |
| D-003 | 2026-10-26 | Category kept only with at least 15 positive test photos from at least 6 scenes | Gives more reliable per-category numbers; fewer categories |
| D-004 | 2026-10-26 | Split dev/test by scene, about 40/60, fixed seed | Photos from one scene are correlated |
| D-005 | 2026-10-26 | Presence-level metrics only | FFS has no box annotations |
| D-006 | 2026-10-26 | Model captions are a stretch goal, not core | Keeps the core focused on detection and evaluation |
| D-007 | 2026-10-26 | Marker finder uses yellow + dark ink, not a strict triangle test | Tent-shaped placards can look like triangles, trapezoids or rectangles |
| D-008 | 2026-10-26 | Find files in the zip by path component names; confirm the layout by running `list` | Real layout could not be inspected before Phase 1. Confirmed the same day (section 4.1) |
| D-010 | 2026-10-26 | One-digit OCR reads are accepted only by exact match, never snapped | Phase 1 demo: a read of "1" (true 12) snapped to 7 with confidence 1.0, because 7 is a confusable one-edit neighbour |
| D-012 | 2026-10-26 | Crop the preprocessed image to the ink box before upscaling | Real EasyOCR failed on a small digit centred in a large white canvas (test_real_ocr_reads_a_synthetic_digit) |
| D-011 | 2026-10-26 | Run OCR on a tight and a wide ink mask and keep the better result | Demo: the second digit sat outside the yellow region and was discarded |
| D-009 | 2026-10-26 | Take a file's scene only from a scene folder or a file-name token, never the archive folder; ignore macOS junk files | First real listing assigned all 40 exhibit lists to th_0001 and counted `._` files as photos |
| D-013 | 2026-10-26 | Finder tuning round 1: H 16-40, S min 150, V min 55, min solidity 0.60, min area fraction 3e-5; ink is relative to the marker body, capped at V 90 (R-A2-8) | Gate failed (H 0.56, A 0.35). th_0006 photos are underexposed: marker body V about 98 (below v_min 110 and the old fixed ink threshold 90), S about 250, H 18-20. Higher S min keeps wood and walls out |

## 10. Change log

| Date | Section | Change |
| --- | --- | --- |
| 2026-10-26 | all | Phase 0: initial spec, sections 1-3 written |
| 2026-10-26 | 4, 8, 9 | Phase 1: marker finder, OCR and exhibit reading requirements; feasibility gate procedure and metrics; D-007, D-008 |
| 2026-10-26 | 4.1, 9 | Layout confirmed from the real zip; fixed scene detection and junk files (R-A0-4, R-A0-5, D-009) |
| 2026-10-26 | 4.1, 4.4, 8.1, 9 | Demo run on the laptop: one-digit snapping banned, two OCR mask variants (R-A3-5, R-A3-6, D-010, D-011); ink-box crop (D-012); part 1 facts (no marker-free scene, two-digit numbers) |
| 2026-10-26 | 4.3, 9 | Gate failed (H 0.56, A 0.35); tuning round 1: relative ink rule R-A2-8, new thresholds in R-A2-2 and R-A2-3, AT-A2-5, AT-A2-6, D-013 |
