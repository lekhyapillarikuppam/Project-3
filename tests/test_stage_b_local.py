"""Acceptance test AT-B-9 (SPEC section 5): the local CPU run writes the same files as the Kaggle run."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stage_b_local as sl  # noqa: E402

N_CLIP = len(sl.sb.flat_queries(sl.sb.CLIP_PROMPTS))
N_OWL = len(sl.sb.flat_queries(sl.sb.OWL_QUERIES))


def fake_clip(img):
    return np.arange(N_CLIP, dtype=float)


def fake_owl(img):
    # Two patches: one strong box in the photo centre, one below the cut-off.
    scores = np.zeros((2, N_OWL))
    scores[0, :] = 0.9
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]])
    return scores, boxes


def photos(n):
    return [{"scene_id": "th_9999", "photo_id": f"P{i}", "path": None} for i in range(n)]


def read(path):
    return list(csv.reader(open(path, encoding="utf-8")))


def test_at_b_9_local_output_format_and_resume(tmp_path):
    load = lambda _: Image.new("RGB", (400, 300))      # noqa: E731
    sl.run(photos(3), tmp_path, "test", fake_clip, fake_owl, load=load, info_extra={"device": "cpu"})

    clip, owl = read(tmp_path / "clip_scores_test.csv"), read(tmp_path / "owlv2_raw_test.csv")
    assert clip[0] == ["scene_id", "photo_id", "category", "prompt", "score"]
    assert owl[0] == ["scene_id", "photo_id", "category", "prompt", "score", "x", "y", "w", "h", "tile"]
    assert len(clip) - 1 == 3 * N_CLIP and len(owl) - 1 == 3 * N_OWL       # one box per query and photo
    assert {r[9] for r in owl[1:]} == {"0"}                                 # tile 0 only
    x, y, w, h = map(float, owl[1][5:9])                                    # box in original pixels
    assert (x, y, w, h) == (160.0, 160.0, 80.0, 80.0)      # boxes are relative to the 400 px padded square
    info = json.loads((tmp_path / "run_info_test.json").read_text(encoding="utf-8"))
    assert info["tiles"] == "tile0 only" and info["device"] == "cpu" and info["complete"]
    assert info["clip_model"] == sl.sb.CLIP_MODEL and info["owl_model"] == sl.sb.OWL_MODEL

    # Resume: a half-written photo is dropped and redone; finished photos are not scored again.
    with open(tmp_path / "clip_scores_test.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["th_9999", "P3", "blade", "a photo of a knife", 1.0])
    seen = []
    sl.run(photos(5), tmp_path, "test", lambda img: (seen.append(1), fake_clip(img))[1], fake_owl, load=load)
    assert len(seen) == 2                                                   # only P3 and P4
    clip = read(tmp_path / "clip_scores_test.csv")
    assert len(clip) - 1 == 5 * N_CLIP
    assert len({r[1] for r in clip[1:]}) == 5
