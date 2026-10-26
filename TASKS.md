# TASKS

Tick items as you go. One phase at a time; do not start a phase before the previous gate passes.
Phases were merged on 2026-10-26 (SPEC D-016).

## Phase 0: Foundation and spec skeleton (done)

- [x] Tools installed, repo created, virtual environment, `pytest` passes, pushed to GitHub
- [x] `SPEC.md` sections 1-3 read

## Phase 1: Feasibility gate (done, D-015)

- [x] Marker finder, OCR, exhibit reader, remote zip fetch
- [x] Real zip layout confirmed (SPEC 4.1)
- [x] Five scenes probed, `truth.csv` hand-labelled (31 photos)
- [x] Two tuning rounds (D-013, D-014); gate passed: H 0.87, A 0.65, precision 0.98 (optimistic: same photos used to tune and test)
- [ ] Confirm license terms; decide how mannequin photos are shown

## Phase 2: Data, categories, split, photo types, markers, silver

- [ ] Merge Phase 2 spec into `SPEC.md` (4.5, 4.6, D-015 to D-017)
- [ ] Tests first: `tests/test_dataprep.py` (AT-P-1 to AT-P-5), `tests/test_silver.py` (AT-S-1 to AT-S-6)
- [ ] Download scenes th_0002 to th_0020; write `exhibits.csv`
- [ ] Run the marker pipeline on every photo; write `markers.csv` (resumable)
- [ ] `category_rules.csv` v1; `unmatched.csv`; confirm categories by hand
- [ ] Exhibit-level category counts; choose final categories (3.2, never re-draw the seed)
- [ ] `scene_split.csv`, `photos.csv` with photo types (check 20 photos by eye)
- [ ] `exhibit_categories.csv` and `silver.csv`
- [ ] List the other parts of the dataset (th_0021 onward)
- **Gate:** all tests pass; at least 4 categories pass the inclusion rule or a logged decision says what to do instead

## Phase 3: Gold labels, models, freeze, test evaluation

- [ ] Labelling screen or notebook; correct silver to gold for all test photos
- [ ] Silver vs gold: precision, recall, kappa per category
- [ ] CLIP baseline prompts and dev scores; OWLv2 prompts and tiling, run once on Kaggle (`owlv2_raw.csv`)
- [ ] Always-present baseline; tune two thresholds per category on dev; box merge
- [ ] Freeze prompts, thresholds, code; tag `protocol-v1`; fill `SPEC.md` sections 5-6
- [ ] Run the frozen pipeline once on test and marker-free scenes
- [ ] Precision, recall, F1, AP per category; macro average; scene-level bootstrap intervals; paired CLIP vs OWLv2
- [ ] Scores against silver and gold; slices by photo type and marker read
- [ ] Marker-masking experiment with random-patch control; false-positive rate on marker-free scenes
- [ ] Error analysis (about 30 FP and 30 FN per method)
- [ ] Export `results.json`, per-scene detections JSON, figures

## Phase 4: Web app and report

- [ ] Static site scaffold; landing with notice, content note, attribution
- [ ] Scene viewer; results page; about and limits
- [ ] Resize photos for the web; choose which to publish
- [ ] Deploy; test the link on a phone
- [ ] Stretch: confirm/reject/edit, scene summary, export, captions, cross-check
- [ ] Finish decisions log and change log in `SPEC.md`
- [ ] README with attribution and license
- [ ] One-paragraph outline for the course evaluation; two-minute spoken summary
