# TASKS

Tick items as you go. One phase at a time; do not start a phase before the previous gate passes.

## Phase 0: Foundation and spec skeleton

- [ ] Install Python 3.11, Git, VS Code, Claude Code (see `docs/PHASE0_SETUP.md`)
- [ ] Unzip the repo and open it in VS Code
- [ ] Create the virtual environment and install `requirements.txt`
- [ ] `python scripts/check_env.py` reports "all good"
- [ ] `pytest` passes
- [ ] Create a GitHub repo and push
- [ ] Read `SPEC.md` sections 1-3 and understand each data file
- **Gate:** all boxes ticked

## Phase 1: Feasibility gate

- [ ] Test pulling only `img/` and `IPC/IPC_ENG/` from the remote Zenodo zip
- [ ] Pick 3-5 scenes that have photos and markers, using the description sheets
- [ ] Open about 20 photos; note marker look, photo quality, mannequins
- [ ] Prototype the colour + shape marker finder; measure hit rate on close-ups (target 70%)
- [ ] Prototype OCR + constrained read; measure accuracy (target 60%)
- [ ] Decide: continue, OWLv2 for markers, or scene-level fallback; log it in `SPEC.md`
- [ ] Confirm license terms; decide how mannequin photos are shown

## Phase 2: Data and categories

- [ ] Download the 30-40 chosen scenes plus 3-5 marker-free scenes
- [ ] Parse exhibit lists into `exhibits.csv`
- [ ] Count recurring items; choose 6-8 categories (15 photos / 6 scenes rule)
- [ ] Write `category_rules.csv`; LLM-suggest the leftovers; confirm by hand
- [ ] Split scenes dev/test by scene (about 40/60, fixed seed); write `scene_split.csv`
- [ ] Tag each photo overview / mid / close; write `photos.csv`

## Phase 3: Stage A, silver labels

- [ ] Run the marker finder on all photos; write `markers.csv`
- [ ] Run OCR and the constrained read
- [ ] Build `silver.csv`; write acceptance tests
- [ ] Fill `SPEC.md` section 4

## Phase 4: Gold labels and detectors on dev

- [ ] Labelling screen or notebook; correct silver to gold for all test photos
- [ ] Silver vs gold: precision, recall, kappa per category
- [ ] CLIP baseline prompts and dev scores
- [ ] OWLv2 prompts, tiling decision; run once on Kaggle; save `owlv2_raw.csv`
- [ ] Always-present baseline
- [ ] Tune two thresholds per category on dev; box merge
- [ ] Freeze prompts, thresholds, code; tag `protocol-v1`; fill `SPEC.md` sections 5-6

## Phase 5: Test evaluation and analysis

- [ ] Run the frozen pipeline once on test and marker-free scenes
- [ ] Precision, recall, F1, AP per category; macro average
- [ ] Scene-level bootstrap (1000) intervals; paired CLIP vs OWLv2
- [ ] Scores against silver and against gold
- [ ] Slices by photo type and marker read
- [ ] Marker-masking experiment with random-patch control
- [ ] False-positive rate on marker-free scenes
- [ ] Error analysis (about 30 FP and 30 FN per method)
- [ ] Export `results.json`, per-scene detections JSON, figures

## Phase 6: Web app, deploy, wrap-up

- [ ] Static site scaffold; landing with notice, content note, attribution
- [ ] Scene viewer; results page; about and limits
- [ ] Resize photos for the web; choose which to publish
- [ ] Deploy; test the link on a phone
- [ ] Stretch: confirm/reject/edit, scene summary, export, captions, cross-check
- [ ] Finish decisions log and change log in `SPEC.md`
- [ ] README with attribution and license
- [ ] One-paragraph outline for the course evaluation; two-minute spoken summary
