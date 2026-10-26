# SPEC: AI-Assisted Evidence Inventory with Marker-Based Ground Truth

Status: Phase 0. Sections 1-3 are written. Sections 4-8 are stubs that get filled
in the phase named next to each one, **before** the code for that phase is written.

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

## 4. Stage A: marker reader and answer key  (filled in Phase 1 and 3)

TBD. Will contain: marker finder requirements, constrained OCR read, silver
label construction, gold correction procedure, and acceptance tests such as:
given a photo where marker 7 is read and exhibit 7 is a knife, silver has
`blade` present for that photo.

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

## 8. Feasibility gate  (filled in Phase 1)

TBD. Pass marks (proposed): marker finder finds at least 70% of markers in
close-ups; OCR plus constrained read is right on at least 60%. Fallbacks: OWLv2
for markers, else scene-level ground truth.

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

## 10. Change log

| Date | Section | Change |
| --- | --- | --- |
| 2026-10-26 | all | Phase 0: initial spec, sections 1-3 written |
