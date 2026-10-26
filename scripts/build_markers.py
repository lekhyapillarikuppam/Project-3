"""Run the final marker pipeline on every photo of every downloaded scene (SPEC R-P-6, R-P-7).

  python scripts/build_markers.py                      # all scenes that have photos
  python scripts/build_markers.py --scenes th_0002     # only these

Needs data_meta/exhibits.csv (run build_exhibits.py first): it gives each scene's valid marker numbers.
Resumable: one part file per scene in results/phase2/markers_parts/. A scene that already has one is
skipped, so new scenes can be added later. After each scene, data_meta/markers.csv and data_meta/photos.csv
are rebuilt from the parts. This script never reads truth.csv and never writes under results/feasibility.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ffs import config, remote_zip as rz  # noqa: E402
from ffs.dataprep import image_size, pending_scenes, photo_types  # noqa: E402
from ffs.markers import MarkerParams, find_markers  # noqa: E402
from ffs.schemas import COLUMNS, validate  # noqa: E402

PARTS = config.RESULTS_DIR / "phase2" / "markers_parts"
PHOTO_PARTS = config.RESULTS_DIR / "phase2" / "photos_parts"


def read_image(path: Path):
    """cv2.imread can fail on non-ASCII Windows paths, so decode from bytes."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def photos_by_scene(raw: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for f in sorted(raw.rglob("*")):
        if f.is_file():
            rel = f.relative_to(raw).as_posix()
            if rz.kind_of(rel) == "photo":
                out.setdefault(rz.scene_of(rel), []).append(f)
    return out


def run_scene(scene: str, photos: list[Path], raw: Path, valid: set[int], reader) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ffs.ocr import read_marker

    params = MarkerParams()
    marker_rows, photo_rows = [], []
    for path in tqdm(photos, desc=scene, file=sys.stdout, mininterval=30):
        img = read_image(path)
        if img is None:
            print(f"  could not read {path.name}")
            continue
        width, height = image_size(path)
        photo_rows.append({"scene_id": scene, "photo_id": path.stem, "relpath": path.relative_to(raw).as_posix(),
                           "width": width, "height": height})
        for i, b in enumerate(find_markers(img, params)):
            res = read_marker(img, b, valid, reader, params)
            marker_rows.append({"scene_id": scene, "photo_id": path.stem, "marker_idx": i,
                                "x": b.x, "y": b.y, "w": b.w, "h": b.h, "method": "color_shape",
                                "det_score": round(b.score, 3), "raw_ocr": res["raw_ocr"],
                                "ocr_conf": res["ocr_conf"],
                                "marker_no": "" if res["marker_no"] is None else int(res["marker_no"]),
                                "status": res["status"]})
    markers = pd.DataFrame(marker_rows, columns=COLUMNS["markers"])
    photo_df = pd.DataFrame(photo_rows, columns=["scene_id", "photo_id", "relpath", "width", "height"])
    return markers, photo_df


def merge_parts(out_dir: Path) -> None:
    """Rebuild markers.csv and photos.csv (with photo_type) from all finished scenes."""
    read = lambda d: [pd.read_csv(f, dtype={"raw_ocr": str, "marker_no": "Int64"}) for f in sorted(d.glob("*.csv"))]
    markers = pd.concat(read(PARTS), ignore_index=True)[COLUMNS["markers"]]
    photos = pd.concat(read(PHOTO_PARTS), ignore_index=True)
    photos["photo_type"] = photo_types(photos, markers)
    photos = photos[COLUMNS["photos"]]
    validate(markers, "markers")
    validate(photos, "photos")
    out_dir.mkdir(parents=True, exist_ok=True)
    markers.to_csv(out_dir / "markers.csv", index=False)
    photos.to_csv(out_dir / "photos.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", help="default: every scene that has photos")
    ap.add_argument("--raw", default=str(config.RAW_DIR))
    ap.add_argument("--meta", default=str(config.META_DIR))
    args = ap.parse_args()
    raw, meta = Path(args.raw), Path(args.meta)

    try:
        exhibits = pd.read_csv(meta / "exhibits.csv")
    except FileNotFoundError:
        sys.exit("data_meta/exhibits.csv is missing. Run scripts/build_exhibits.py first.")
    valid = {s: set(g["marker_no"]) for s, g in exhibits.groupby("scene_id")}

    found = photos_by_scene(raw)
    wanted = [s.lower() for s in args.scenes] if args.scenes else sorted(found)
    todo = pending_scenes(wanted, PARTS)
    print(f"{len(wanted)} scenes with photos, {len(wanted) - len(todo)} already done, {len(todo)} to run", flush=True)

    reader = None
    if todo:
        from ffs import ocr
        reader = ocr.get_reader()
    PARTS.mkdir(parents=True, exist_ok=True)
    PHOTO_PARTS.mkdir(parents=True, exist_ok=True)
    for scene in todo:
        if scene not in found:
            print(f"{scene}: no photos under {raw}, skipped", flush=True)
            continue
        markers, photos = run_scene(scene, found[scene], raw, valid.get(scene, set()), reader)
        photos.to_csv(PHOTO_PARTS / f"{scene}.csv", index=False)
        markers.to_csv(PARTS / f"{scene}.csv", index=False)     # written last: its existence means "done"
        merge_parts(meta)
        print(f"{scene}: {len(photos)} photos, {len(markers)} boxes", flush=True)

    if list(PARTS.glob("*.csv")):
        merge_parts(meta)
    done = pd.read_csv(meta / "markers.csv")
    print(f"FINISHED: {pd.read_csv(meta / 'photos.csv').shape[0]} photos, {len(done)} markers in {meta}", flush=True)


if __name__ == "__main__":
    main()
