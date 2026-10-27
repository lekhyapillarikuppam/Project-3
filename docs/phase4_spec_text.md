# Text to add to SPEC.md (Phase 3 end)

Decision row (section 9):
| D-031 | 2026-10-27 | Human review covers only the AI-vs-silver disagreements (`make_gold_tool.py --review --only-disagree`); low-confidence-only pairs keep the AI label | The author's time is short. The 60-photo audit still measures the AI labeller's error on confident, agreed photos; low-confidence labels not checked by a human are reported as a limitation |
| D-032 | 2026-10-27 | Marker-masking experiment, photo-type slices and tiling on the test split are dropped | Time. Dev showed no gain from tiling. Scope of claims in the report is narrowed accordingly |
| D-033 | 2026-10-27 | Test split run on the author's CPU (tile 0 only) if Kaggle is not used | Same models, prompts and thresholds as the frozen protocol; only the device differs |

Section 6, acceptance tests: AT-E-1 average precision on known rankings; AT-E-2 precision/recall/F1/kappa; AT-E-3 CLIP photo score; AT-E-4 OWLv2 photo score and best box (tile 0 only);
AT-E-5 thresholds found in the protocol file; AT-E-6 reproducible scene bootstrap and paired difference; AT-E-7 whole scenes are resampled; AT-E-8 `evaluate_test.py` end to end;
AT-E-9 `--only-disagree` filter.

Requirements:
- R-E-1. `scripts/evaluate_test.py` reads thresholds only from `protocol/protocol_v1.json` and refuses to run without the git tag `protocol-v1`.
- R-E-2. Main reference is gold (test split). Silver is shown as a second reference, and silver judged against gold answers "how trustworthy is the automatic answer key".
- R-E-3. Intervals: scene-level cluster bootstrap, 1000 resamples, seed 42, percentile 95%. OWLv2 vs CLIP is a paired difference on the same resamples.
