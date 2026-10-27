"""AT-A-1..2: the static demo site is built from saved results and works in a browser."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ffs import config

CATS = list(config.FINAL_CATEGORIES)


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_app.py"
    spec = importlib.util.spec_from_file_location("build_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def project(tmp_path, monkeypatch):
    meta, res, raw = tmp_path / "data_meta", tmp_path / "results" / "eval", tmp_path / "raw"
    for d in (meta, res, raw):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    photos = [(f"s{s}", f"p{p}") for s in range(6) for p in range(5)]
    for s, p in photos:
        Image.new("RGB", (1600, 1200), (90, 90, 90)).save(raw / f"{s}_{p}.jpg")
    pd.DataFrame({"scene_id": [a for a, _ in photos], "photo_id": [b for _, b in photos],
                  "relpath": [f"{a}_{b}.jpg" for a, b in photos], "width": 1600, "height": 1200,
                  "photo_type": "close"}).to_csv(meta / "photos.csv", index=False)
    rows = []
    for s, p in photos:
        for c in CATS:
            g = int(rng.random() < 0.3)
            rows.append((s, p, c, 0.2, int(g), 0.5, g, "q", 200.0, 100.0, 400.0, 300.0, int(rng.random() < 0.2), g))
    pd.DataFrame(rows, columns=["scene_id", "photo_id", "category", "clip_score", "clip_pred", "owl_score", "owl_pred",
                                "owl_prompt", "box_x", "box_y", "box_w", "box_h", "silver", "gold"]).to_csv(
        res / "photo_predictions.csv", index=False)
    m = [("gold", sy, c, 10, .5, .5, .5, .3, .7, .5, .3, .7) for sy in ("CLIP", "OWLv2") for c in CATS + ["macro"]]
    pd.DataFrame(m, columns=["reference", "system", "category", "n_pos", "precision", "recall", "f1", "f1_lo", "f1_hi",
                             "ap", "ap_lo", "ap_hi"]).to_csv(res / "test_metrics.csv", index=False)
    pd.DataFrame([(c, 10, .9, .5, .6, .4, .8, .5) for c in CATS + ["macro"]],
                 columns=["category", "n_pos", "precision", "recall", "f1", "f1_lo", "f1_hi", "kappa"]).to_csv(
        res / "silver_vs_gold.csv", index=False)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    return tmp_path


def test_at_a_1_build_is_deterministic_and_has_positives_of_every_category(project):
    mod = _load()
    out1, out2 = project / "site1", project / "site2"
    n1 = mod.build(project, 12, out1)
    mod.build(project, 12, out2)
    assert n1 == 12 and (out1 / "data.js").read_text() == (out2 / "data.js").read_text()
    data = json.loads((out1 / "data.js").read_text().split("=", 1)[1].rstrip().rstrip(";"))
    for c in CATS:
        assert any(p["cats"][c]["gold"] for p in data["photos"]), c
    for p in data["photos"]:
        assert (out1 / p["img"]).exists() and max(Image.open(out1 / p["img"]).size) <= 1000
        for r in p["cats"].values():
            assert r["box"] is None or all(0 <= v <= 1 for v in r["box"])
    assert {"index.html", "data.js", ".nojekyll"} <= {f.name for f in out1.iterdir()}


def test_at_a_2_site_runs_in_a_browser_without_errors(project):
    pw = pytest.importorskip("playwright.sync_api")
    mod = _load()
    out = project / "site"
    mod.build(project, 10, out)
    errors = []
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 390, "height": 800})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((out / "index.html").as_uri())
        assert page.locator(".tile").count() == 10
        page.locator(".tile").first.click()
        assert page.locator("#det table").count() == 1
        assert page.locator(".box").count() >= 0
        page.click("#t-res")
        assert "OWLv2" in page.inner_text("#res") and "answer key" in page.inner_text("#res")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        b.close()
    assert not errors
