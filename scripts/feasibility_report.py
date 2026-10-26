"""Compute the feasibility-gate metrics and verdict (SPEC 8.2 and 8.3).

  python scripts/feasibility_report.py

Needs results/feasibility/truth.csv with the two blank columns filled in for the
photos you checked: n_markers_visible, true_marker_nos (numbers separated by ;).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config  # noqa: E402
from ffs.feasibility import MIN_FINDER_HIT, MIN_READ_ACCURACY, compute_metrics, verdict  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(config.FEASIBILITY_DIR))
    args = ap.parse_args()
    d = Path(args.dir)

    try:
        truth = pd.read_csv(d / "truth.csv", dtype={"true_marker_nos": str})
        photos = pd.read_csv(d / "probe_photos.csv", dtype={"read_nos": str})
        markers = pd.read_csv(d / "probe_markers.csv", dtype={"raw_ocr": str})
    except FileNotFoundError as e:
        sys.exit(f"Missing file: {e.filename}. Run scripts/run_marker_probe.py first.")

    n_labelled = int(truth["n_markers_visible"].notna().sum())
    if n_labelled == 0:
        sys.exit("truth.csv has no labelled rows yet. Fill n_markers_visible and true_marker_nos.")
    print(f"Labelled photos: {n_labelled} of {len(truth)}")

    m = compute_metrics(truth, photos, markers)
    lines = [
        f"Photos labelled:                {m['n_photos']} ({m['n_photos_with_markers']} with markers, "
        f"{m['n_true_markers']} markers in total)",
        f"Finder hit rate H:              {m['finder_hit_rate']:.2f}   (pass mark {MIN_FINDER_HIT:.2f})  "
        f"{'PASS' if m['pass_finder'] else 'FAIL'}",
        f"Finder false boxes per photo:   {m['finder_fp_per_photo']:.2f}",
        f"Read accuracy (constrained) A:  {m['read_accuracy']:.2f}   (pass mark {MIN_READ_ACCURACY:.2f})  "
        f"{'PASS' if m['pass_read'] else 'FAIL'}",
        f"Read accuracy (raw OCR only):   {m['raw_read_accuracy']:.2f}",
        f"Precision (report only):        {m['precision']:.2f}   (aim 0.90)",
        "",
        "Per scene:",
        m["per_scene"].round(2).to_string(index=False),
        "",
        verdict(m),
    ]
    text = "\n".join(lines)
    print("\n" + text)
    (d / "report.txt").write_text(text, encoding="utf-8")
    print(f"\nSaved {d / 'report.txt'}")


if __name__ == "__main__":
    main()
