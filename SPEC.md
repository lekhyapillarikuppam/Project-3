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
| `results/clip_scores_<split>.csv` | photo x category x prompt | `scene_id, photo_id, category, prompt, score` |
| `results/owlv2_raw_<split>.csv` | raw box | `scene_id, photo_id, category, prompt, score, x, y, w, h, tile` |

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

`clip_scores` and `owlv2_raw` are written per split: `results/clip_scores_dev.csv`, `results/clip_scores_test.csv`,
`results/owlv2_raw_dev.csv`, `results/owlv2_raw_test.csv` (D-022). Columns are unchanged.

---

## 3. Category definitions

### 3.1 Candidate categories

The final categories are chosen in Phase 2 by the inclusion rule below (decision D-020); the table lists the candidates.

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

The rule is applied mechanically to the silver table of candidate categories (`build_split_silver.py`), once, and the result is `FINAL_CATEGORIES` in `config.py` (D-020). Categories that fail are not dropped from the data: their exhibits keep their candidate label in `exhibit_categories.csv`, but they are merged into `other` for all per-category metrics. The split seed is never re-drawn to make a category pass (D-017). The silver counts are an estimate; Phase 3 gold labels on the test split may move a category across the line, and then D-020 is revisited with a logged decision.

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

## 4. Stage A: marker reader and answer key  (4.1-4.4 written in Phase 1; 4.5-4.6 in Phase 2; 4.7 in Phase 3)

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
- R-A3-7. Rotation retries (D-014a). If the best read so far is `rejected` or has `ocr_conf` below 0.60,
  both preprocessed images are also rotated by +30, -30, +60 and -60 degrees (white fill, canvas grown so
  nothing is cut) and read again. All reads compete under R-A3-6.
- R-A3-8. (Tried in D-014b and **reverted**: scaling crops under 120 px to at least 200 px tall instead of
  140 changed neither H, A nor precision. Not part of the pipeline.)
- R-A3-9. Close-ups (D-014c). Ink pieces shorter than 30% of the tallest ink piece are dropped before
  the ink-box crop (this removes the ruler strip printed on the marker and specks). After the crop the
  image is scaled down, never up, so it is at most 220 px tall. Evidence: EasyOCR misread the 1,000 px
  wide close-ups when given the full-size image.

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
- AT-A3-12. A synthetic marker rotated 50 degrees (plain EasyOCR reads it as "4") is read correctly by `read_marker` (rotation retries, R-A3-7).
- AT-A3-13. A synthetic marker with a ruler strip, drawn 1,200 px wide, is preprocessed to at most 220 px
  tall with only the digit pieces left (R-A3-9), and is read correctly by `read_marker`.
- AT-A3-14. (Removed with R-A3-8, see D-014b.)

### 4.5 Silver labels (gold is Phase 3)

Silver = the answer key built automatically from markers and exhibit lists.

- R-S-1. A marker is **counted** if its `status` is `accepted` or `snapped` and its `marker_no` is in
  that scene's exhibit list. Rejected boxes are never counted.
- R-S-2. For a photo and a final category c: `present = 1` if at least one counted marker in the photo
  belongs to an exhibit of category c (`exhibit_categories.csv`, rule rows and confirmed llm/manual rows only);
  otherwise `present = 0`. `marker_nos` lists the counted numbers that made it 1.
- R-S-3. Silver is a full matrix: every photo has one row for every final category (not `other`).
- R-S-4. If a scene's list repeats a number (th_0014: 15 twice), the marker maps to all of those
  exhibits: the union of their categories is present.
- R-S-5. An **uncertain photo** is one with at least one rejected box. It is not stored in `silver.csv`;
  it is computed from `markers.csv` and used as an analysis slice (silver quality on certain vs
  uncertain photos).
- R-S-6. Known limits of silver (these are what Phase 3 measures against gold): it misses evidence
  that has no marker, markers the finder or reader missed (read accuracy was 0.65 on the gate sample,
  so expect high precision and moderate recall), and exhibits whose marker is visible but the object is
  out of frame. Silver is **not** truth; it is an estimate to be checked against hand-corrected gold.
- R-S-7. `silver.csv` holds the final categories only (`config.FINAL_CATEGORIES`), as a full photo x category matrix
  (R-S-3). The same table for all candidate categories is `results/phase2/silver_candidates.csv` (not committed); it is
  the basis of the inclusion table in D-020.

Acceptance tests (`tests/test_silver.py`):

- AT-S-1. Marker 7 read as accepted in a photo, exhibit 7 is "knife": `blade` present with `marker_nos = "7"`;
  every other category 0.
- AT-S-2. A rejected box adds no label.
- AT-S-3. A scene list with number 15 twice (knife, bottle): a counted marker 15 sets both `blade` and `bottle_glass`.
- AT-S-4. The output has exactly (photos x final categories) rows and passes `validate(df, "silver")`.
- AT-S-5. A counted-looking marker whose number is not in the scene's list is ignored.
- AT-S-6. A photo with no markers has all categories 0 and is not uncertain; a photo with one rejected box is uncertain.

### 4.6 Data preparation (Phase 2)

- R-P-1. `exhibits.csv`: all English exhibit lists of scenes th_0002 to th_0020 (th_0001 has a list but no
  photos; keep its rows but it is not split). Read with `exhibits.read_exhibit_file`. Rows with an empty
  marker number are dropped and counted in a printed report. Repeated numbers are kept.
- R-P-2. `category_rules.csv` (version 1): case-insensitive regex patterns on the English description,
  from the candidate list in section 3.1 and from what the descriptions really contain. Rules are
  **multi-label** (D-018): every matching rule adds its category to the exhibit, so "knife with reddish
  trace" is both `blade` and `red_brown_stain`. Rules whose category is `other` (for example "glass table",
  body words) are **exclusive**: if one matches, the exhibit is `other` only. `exhibit_categories.csv` has one
  row per exhibit and category. Rebuilding it keeps the existing `llm` and `manual` rows of exhibits that still
  match no rule, with their `confirmed` value, so confirmations are never lost.
  Descriptions matching no rule are listed in `results/phase2/unmatched.csv`; Claude Code proposes a category
  for each (`source = llm`, `confirmed = 0`). Only rows confirmed by the user (`confirmed = 1`) or rule rows count;
  unconfirmed llm rows are treated as `other`.
- R-P-3. Category counts are reported at exhibit level (exhibits and scenes per category) **before** any
  model is run. Final categories follow the inclusion rule of section 3.2, checked on the test split once
  silver exists. If fewer than 4 categories pass, the options are: add scenes from more parts of the dataset,
  merge categories, or lower the bar with wider reported intervals. Choose one and log it as a decision.
  **Never re-draw the split seed to make a category pass.**
- R-P-4. `scene_split.csv`: only scenes that have photos. Shuffle with `config.SEED` (42), the first
  `round(DEV_FRACTION * n)` scenes are `dev`, the rest `test`. Same seed gives the same split. `has_markers` is
  1 if the scene's list has at least one marker. `n_photos` counts photos.
- R-P-5. `photo_type` from the largest found marker box in the photo, as a fraction of photo height:
  `close` if at least `CLOSE_MIN_FRAC` (0.30), `mid` if at least `MID_MIN_FRAC` (0.10), else `overview`;
  `unknown` if no box was found. Thresholds live in `config.py`; any change is logged. Check 20 random
  photos by eye when first built and record the result in the decision log.
- R-P-6. `markers.csv` for all photos of the scenes in `scene_split.csv`, with the final pipeline (finder +
  tight/wide OCR + rotation retries + constrained read against that scene's list), method `color_shape`.
  The script is resumable: a scene already done is skipped, so new scenes can be added later.
- R-P-7. `photos.csv`: width and height read from the image header, `relpath` under `data/raw/`.

Acceptance tests (`tests/test_dataprep.py`):

- AT-P-1. Rule matching: "knife with black handle" -> `blade`; "half-empty bottle of vodka" -> `bottle_glass`;
  "mobile phone" -> `phone_electronics`; an unlisted word -> no match.
- AT-P-2. The split is by scene, disjoint, deterministic for a seed, and about 40/60.
- AT-P-3. Photo type thresholds: box height fraction 0.35 -> close, 0.15 -> mid, 0.03 -> overview, no box -> unknown.
- AT-P-4. `exhibits.csv` drops rows with empty marker numbers and keeps repeated numbers.
- AT-P-5. The marker script skips a scene that is already done.
- AT-P-6. Multi-label rules: "Knife with reddish trace" gives `blade` and `red_brown_stain`; "Glass table"
  gives `other` only (exclusive rule), although "glass" alone would match `bottle_glass`.

### 4.7 Gold labels (Phase 3)

Gold = the answer key the project trusts: the same photo x category matrix as silver, filled by a person
looking at the photo.

- R-G-1. Gold is made for **all test photos** (545) and every final category. Dev photos get no gold.
- R-G-2. **Blind labelling.** The labelling page never shows silver. The exhibit list of the scene is hidden
  and opens only on request (key L), so the labeller can peek when unsure but is not led by it. Marker
  numbers on the photo may be read like any other part of the photo.
- R-G-3. Definition. `present = 1` if at least one object of the category is clearly recognisable in the
  photo, even if small or partly hidden, whether or not it has a marker or is on the exhibit list.
  ammunition: cartridge cases, bullets, shotgun shells. blade: knife, cutter, scissors, axe, saw, razor.
  bottle_glass: bottles, drinking glasses, vases, glass containers, broken glass. paper_document: sheets,
  notebooks, books, newspapers, envelopes, letters, ID. red_brown_stain: a red-brown stain or trace on
  any surface or object (never called blood). Not recognisable even when zoomed: 0.
- R-G-4. Completeness. `data_meta/gold.csv` is written only when every test photo has a label for every
  final category (`scripts/import_gold.py`; `--allow-partial` is for progress checks and writes nothing with `--dry-run`).
- R-G-5. `corrected = 1` when gold differs from silver for that photo and category. It is computed by the
  import script, not by the labelling page.
- R-G-6. Reliability. One annotator (the author). A second blind pass over 55 random test photos
  (`make_gold_tool.py --pass2 55`), done at least a day after pass 1, gives per-category agreement and
  Cohen's kappa (`import_gold.py --agreement`). A category with kappa below 0.70 is flagged in the results.
- R-G-7. Silver-vs-gold report (`scripts/silver_vs_gold.py`, results in `results/silver_vs_gold.csv`):
  per category, silver against gold on the test split: TP, FP, FN, TN, precision, recall, F1 and Cohen's
  kappa, each with a 95% interval from a **scene-level cluster bootstrap** (resample scenes with replacement,
  1000 resamples, seed `config.SEED`). Also the gold prevalence (share of test photos with the category).
  Slices: by `photo_type`; certain vs uncertain photos (R-S-5); photos with at least one counted marker
  vs none.
- R-G-8. Reading the result: silver precision shows how often a marker-based label is wrong; silver recall
  shows what the markers miss (objects without markers, unread markers). This is the project's answer to
  "how trustworthy is the answer key".

Acceptance tests:

- AT-G-1 to AT-G-6: `tests/test_gold.py` (tool data has test photos only and no silver; pass 2 sample is
  deterministic; import marks corrections; incomplete labels are refused; kappa; intra-rater agreement).
- AT-G-7. Silver-vs-gold on a hand-computed example (for instance 10 photos, one category) gives the exact
  TP, FP, FN, TN, precision, recall, F1 and kappa.
- AT-G-8. The scene bootstrap is deterministic for a seed and resamples whole scenes (a photo never
  appears without its scene-mates); with every photo in one scene per resample the interval collapses sensibly.
- AT-G-9. Slice rows add up to the overall counts.

## 5. Stage B: detectors

Two zero-shot models score the photos. Neither ever sees silver, gold or exhibit lists (hard rule).

- R-B-1. **CLIP baseline** (`openai/clip-vit-large-patch14`): the whole photo, padded to a square (nothing
  is cropped away), against 3 text prompts per category. Score = image-text logit (cosine x 100). Photo-level
  presence = max over the category's prompts above a per-category threshold (tuned on dev). Saved raw.
- R-B-2. **OWLv2 main** (`google/owlv2-base-patch16-ensemble`): text queries per category (short noun phrases).
  Each photo is run as the full photo (tile 0) and as 2 x 2 overlapping tiles (tiles 1 to 4, 25% overlap),
  so small objects get more pixels. Boxes are mapped back to **original photo pixels**. All boxes with
  score at least 0.05 are kept, at most 20 per photo, query and tile. Photo-level presence = best box score
  of the category above a per-category threshold (tuned on dev).
- R-B-3. Raw outputs are saved once per split and thresholds are tuned afterwards on a laptop, so models are
  not re-run for tuning. The tile column lets dev decide whether tiling helps (use tile 0 only vs all tiles).
- R-B-4. The code is `kaggle/stage_b_run.py` (one file, run on a Kaggle GPU) and writes
  `clip_scores_<split>.csv`, `owlv2_raw_<split>.csv` and `run_info_<split>.json` (models, prompts, settings,
  library versions). Prompts never use the word "blood".
- R-B-5. Photos are shrunk to at most 2400 px on the long side for upload (`scripts/package_stage_b.py`);
  boxes are scaled back using the original size in the manifest.
- R-B-6. **Dev first.** Prompts and thresholds are tuned on dev only. **Test photos are packaged and scored
  only after the git tag `protocol-v1` exists**; `package_stage_b.py --split test` refuses otherwise.
- R-B-7. `scripts/import_stage_b.py` checks the returned CSVs against the data contracts and that every
  photo of the split has CLIP scores.

Acceptance tests (`tests/test_stage_b.py`): AT-B-1 tiles cover the photo and overlap; AT-B-2 boxes map back to
original pixels; AT-B-3 boxes in the padding are dropped and others clamped; AT-B-4 top-k limit;
AT-B-5 prompts avoid "blood" and cover the same categories for both models; AT-B-6 test packaging needs the
freeze tag and shrinking keeps the original size.

## 6. Evaluation protocol  (filled in Phase 3, frozen by git tag `protocol-v1`)

TBD. Planned content (from the project plan): scene-level dev/test split
(about 40/60, fixed seed); per-category precision, recall, F1 and AP; macro
average; scene-level bootstrap with 1000 resamples for 95% intervals and a
paired CLIP vs OWLv2 comparison; always-present baseline; silver vs gold
agreement (precision, recall, Cohen's kappa); scores against silver and gold;
slices by photo type and marker read; marker-masking experiment with a
random-patch control; false-positive rate on marker-free scenes; error analysis
of about 30 false positives and 30 false negatives per method.

### 6.1 Freeze checklist

Before the tag `protocol-v1` is created, these are written down and committed: final categories (D-020), prompts
and queries, the tiling choice, per-category thresholds for CLIP and OWLv2, the box merge rule, the photo-level
decision rule, the metric definitions, and the code commit. After the tag, nothing about them changes. The test
split is scored once. Any later change is a new, clearly labelled experiment, not part of the main result.

## 7. Web app  (filled in Phase 4)

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
- **Precision** (report only, not a gate) = among markers with status `accepted` or `snapped` in labelled
  photos, the share whose `marker_no` is in that photo's `true_marker_nos`. It shows how often a number
  the pipeline commits to is wrong. Aim for at least 0.90, flag it if lower.

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
| D-014 | 2026-10-26 | Tuning round 2 (the last): (a) rotation retries R-A3-7, (b) min height 200 for small markers R-A3-8, (c) close-ups: drop short ink pieces and cap the OCR image at 220 px R-A3-9, (d) only if A stays below 0.60: fix small-blob rejection and split big blobs. Each change is kept only if it raises A or H and precision (report-only metric, SPEC 8.2) falls by at most 0.05. **Outcome** (H / A / precision / false boxes per photo; start 0.87 / 0.49 / 1.00 / 0.23): (a) kept, 0.87 / 0.57 / 1.00 / 0.23; (b) **reverted**, no change at all (0.87 / 0.57 / 1.00 / 0.23), R-A3-8 and AT-A3-14 removed; (c) kept, 0.87 / 0.65 / 0.98 / 0.23; (d) **not done**, because A reached 0.60 after (c). The small-blob ink-fraction rejection (th_0011 100604) is therefore still open | Evidence: close-ups th_0002 085557/085623/085653 have correct boxes and sharp digits but the 1,500 px preprocessed image (ruler strip included) gave "2", "" and "7"; shrinking to 220 px gave 12, 14 and 17 (conf 0.89-1.00). th_0011 100709 and 100552: the "12" is on the side face of a tent marker, rotated and sheared. th_0011 100604 "32" and "19": 33-43 px blobs rejected only by the ink fraction (0.014 and 0.000, minimum 0.02). **Caveat: tuning and evaluation use the same 31 hand-labelled photos, so the gate numbers are optimistic.** |
| D-009 | 2026-10-26 | Take a file's scene only from a scene folder or a file-name token, never the archive folder; ignore macOS junk files | First real listing assigned all 40 exhibit lists to th_0001 and counted `._` files as photos |
| D-013 | 2026-10-26 | Finder tuning round 1: H 16-40, S min 150, V min 55, min solidity 0.60, min area fraction 3e-5; ink is relative to the marker body, capped at V 90 (R-A2-8) | Gate failed (H 0.56, A 0.35). th_0006 photos are underexposed: marker body V about 98 (below v_min 110 and the old fixed ink threshold 90), S about 250, H 18-20. Higher S min keeps wood and walls out |
| D-015 | 2026-10-26 | Phase 1 gate passed: continue with colour + shape finder and EasyOCR (H 0.87, A 0.65, precision 0.98 on 31 photos, optimistic because tuned and tested on the same photos) | After D-013 and D-014 |
| D-016 | 2026-10-26 | Phases 2 and 3 merged into one Phase 2 (data, categories, split, photo types, markers on all photos, silver); old phases 4+5 merged into Phase 3 (gold, Kaggle models, freeze, test evaluation); web app is Phase 4 | Speed; the user asked to combine phases |
| D-017 | 2026-10-26 | Split seed is never re-drawn to make a category pass the inclusion rule | Protects the evaluation from seed shopping |
| D-018 | 2026-10-26 | Category rules are multi-label; `other` rules are exclusive | First-match-wins lost the stain on "knife with reddish trace" (th_0003, th_0009); the user asked for multi-label. Exclusive `other` keeps "glass table" and bodies out of `bottle_glass` |
| D-019 | 2026-10-26 | Use all four parts of the FFS dataset on Zenodo (V 0.1, all CC BY-NC-SA 4.0, access open): part 1 record 17151982 `th_0001-0020.zip`, part 2 record 17164106 `th_0021-0040.zip` (42.3 GB), part 3 record 17164834 `th_0041-0060.zip` (40.1 GB), part 4 record 17165128 `th_0061_0080.zip` (47.5 GB). Only photos, English exhibit lists and description sheets are fetched, with `fetch_scenes.py --part N`. The dataset pages say newer versions exist; they were **not** used. Part 3 has no scenes 45, 47, 49, 51, 53; part 4's top folder is `th_0061_0080` | The merged data gives more scenes per category (D-003 inclusion rule needs 6 test scenes). Licences read from the Zenodo API on the same day; same share-alike terms as part 1, so the project keeps CC BY-NC-SA 4.0 for derived data |
| D-020 | 2026-10-26 | **Final categories** (mechanical, rule of 3.2 on the silver test split, seed 42: 26 dev and 40 test scenes, 350 dev and 545 test photos): `ammunition`, `blade`, `bottle_glass`, `paper_document`, `red_brown_stain`. Merged into `other` for per-category metrics: `firearm`, `footwear`, `phone_electronics` | Test positives (photos / scenes): ammunition 43 / 7, blade 19 / 6, bottle_glass 59 / 13, paper_document 52 / 14, red_brown_stain 87 / 13 pass; firearm 20 / 4, footwear 10 / 4, phone_electronics 13 / 6 fail (15 photos and 6 scenes needed). Dev positives (photos / scenes): ammunition 45 / 5, blade 8 / 3, bottle_glass 46 / 7, firearm 25 / 3, footwear 5 / 2, paper_document 32 / 9, phone_electronics 13 / 5, red_brown_stain 40 / 7. Exhibit-level counts (exhibits / scenes, rule and confirmed rows, 531 exhibits in 75 scenes): blade 16 / 13, firearm 10 / 9, ammunition 38 / 17, bottle_glass 49 / 26, phone_electronics 15 / 14, paper_document 35 / 28, footwear 9 / 8, red_brown_stain 43 / 22, other 321 / 74. `blade` passes only just (6 scenes). Five categories pass, so the "fewer than 4" options of R-P-3 are not needed |
| D-021 | 2026-10-26 | No third finder tuning round; the known gaps are accepted and measured instead | Known gaps: th_0005 markers have body V 22 to 50, below `v_min` 55; th_0008 has photos with a blue colour cast (marker body H about 130, S about 67); many dark photos give `unknown` photo type. Silver recall is therefore expected to be low in those scenes. Gold correction of every test photo (Phase 3) fixes the test side; dev is not corrected. No scene with photos is marker-free in parts 1 to 4 (the sheet rows with zero markers also have zero photographs: th_0039, th_0041, th_0045, th_0047, th_0049, th_0051, th_0053, th_0080), so the false-positive check uses photos in which no marker is visible, plus the marker-masking experiment |
| D-022 | 2026-10-27 | Stage B outputs are one file per split (dev, test) | Test is scored only after the freeze |
| D-023 | 2026-10-27 | Gold labelling is blind (silver hidden, exhibit list on request), one annotator, with a blind second pass on 55 photos | Avoids anchoring on silver; reliability measured with kappa |
| D-024 | 2026-10-27 | CLIP baseline sees the whole photo padded to a square; OWLv2 sees the photo plus 2x2 tiles; tiling decided on dev from the saved tile column | Fair baseline; small objects need more pixels |
| D-025 | 2026-10-27 | Models: CLIP ViT-L/14 and OWLv2 base ensemble; photos shrunk to 2400 px for upload | Fits a free Kaggle GPU; boxes mapped back to original pixels |

## 10. Change log

| Date | Section | Change |
| --- | --- | --- |
| 2026-10-26 | all | Phase 0: initial spec, sections 1-3 written |
| 2026-10-26 | 4, 8, 9 | Phase 1: marker finder, OCR and exhibit reading requirements; feasibility gate procedure and metrics; D-007, D-008 |
| 2026-10-26 | 4.1, 9 | Layout confirmed from the real zip; fixed scene detection and junk files (R-A0-4, R-A0-5, D-009) |
| 2026-10-26 | 4.1, 4.4, 8.1, 9 | Demo run on the laptop: one-digit snapping banned, two OCR mask variants (R-A3-5, R-A3-6, D-010, D-011); ink-box crop (D-012); part 1 facts (no marker-free scene, two-digit numbers) |
| 2026-10-26 | 4.3, 9 | Gate failed (H 0.56, A 0.35); tuning round 1: relative ink rule R-A2-8, new thresholds in R-A2-2 and R-A2-3, AT-A2-5, AT-A2-6, D-013 |
| 2026-10-26 | 4.4, 8.2, 9 | Tuning round 2: R-A3-7 and R-A3-9 (R-A3-8 tried and reverted), AT-A3-12 and AT-A3-13, report-only precision metric, D-014 |
| 2026-10-26 | 4.5, 4.6, 9 | Phase 2 spec merged: silver labels (R-S-1 to R-S-6, AT-S-1 to AT-S-6), data preparation (R-P-1 to R-P-7, AT-P-1 to AT-P-5), D-015 to D-017; phases renumbered (D-016) |
| 2026-10-26 | 4.6, 9 | Category rules multi-label with exclusive `other` rules (R-P-2, AT-P-6, D-018); llm suggestions confirmed except th_0002 #16 (manual, other) |
| 2026-10-26 | 4.1, 9 | Parts 2-4 of the dataset added (config URLs, `fetch_scenes.py --part` after the command, D-019) |
| 2026-10-26 | 3.1, 3.2, 4.5, 9 | Phase 2 finished: final categories chosen mechanically (D-020, `config.FINAL_CATEGORIES`), `silver.csv` limited to them (R-S-7), no third tuning round (D-021); stray heading from the merged text removed |
| 2026-10-27 | 2, 4.7, 5, 6.1, 9 | Phase 3 spec merged: gold labels (R-G-1 to R-G-8, AT-G-1 to AT-G-9), Stage B (R-B-1 to R-B-7, AT-B-1 to AT-B-6), freeze checklist, per-split Stage B files, D-022 to D-025 |
