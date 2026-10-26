"""Run the marker finder (and OCR) on downloaded scenes (SPEC 8.1, step 3).

  python scripts/run_marker_probe.py --scenes th_0003 th_0007
  python scripts/run_marker_probe.py --scenes th_0003 --no-ocr        # finder only, fast

Writes to results/feasibility/:
  probe_markers.csv   one row per marker box (raw read, constrained read, status)
  probe_photos.csv    one row per photo (how many boxes, which numbers were read)
  overlays/           photos with boxes and reads drawn on them: LOOK AT THESE
  truth.csv           blank sheet for you to fill in (never overwritten unless --force)
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
from ffs.exhibits import read_exhibit_file  # noqa: E402
from ffs.markers import MarkerParams, draw_overlay, find_markers  # noqa: E402


def read_image(path: Path):
    """cv2.imread can fail on non-ASCII Windows paths, so decode from bytes."""
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def scene_files(raw: Path, scene: str, kind: str) -> list[Path]:
    out = []
    for f in sorted(raw.rglob("*")):
        if f.is_file():
            rel = f.relative_to(raw).as_posix()
            if rz.kind_of(rel) == kind and rz.scene_of(rel) == scene:
                out.append(f)
    return out


def valid_numbers(raw: Path, scene: str) -> set[int]:
    nums: set[int] = set()
    for f in scene_files(raw, scene, "exhibit_list"):
        try:
            nums |= set(read_exhibit_file(f)["marker_no"].tolist())
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: could not read {f.name}: {e}")
    return nums


def pick_sample(photos: pd.DataFrame, per_scene: int) -> pd.DataFrame:
    """Evenly spread sample: mostly photos with boxes, plus up to 2 with none."""
    chosen = []
    for _, g in photos.groupby("scene_id"):
        with_boxes = g[g.n_boxes > 0].sort_values("photo_id")
        without = g[g.n_boxes == 0].sort_values("photo_id")
        n_without = min(len(without), 2)
        n_with = min(len(with_boxes), per_scene - n_without)
        pick = lambda d, n: d.iloc[np.unique(np.linspace(0, len(d) - 1, n).round().astype(int))] if n else d.iloc[0:0]
        chosen += [pick(with_boxes, n_with), pick(without, n_without)]
    return pd.concat(chosen) if chosen else photos.iloc[0:0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--raw", default=str(config.RAW_DIR))
    ap.add_argument("--out", default=str(config.FEASIBILITY_DIR))
    ap.add_argument("--no-ocr", action="store_true", help="only find boxes; skip OCR")
    ap.add_argument("--max-photos", type=int, help="limit photos per scene (quick test)")
    ap.add_argument("--sample-per-scene", type=int, default=8, help="rows per scene in truth.csv")
    ap.add_argument("--force", action="store_true", help="overwrite an existing truth.csv")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    params = MarkerParams()

    reader = None
    if not args.no_ocr:
        try:
            from ffs import ocr
            reader = ocr.get_reader()
        except ImportError:
            sys.exit("EasyOCR is not installed. Run: pip install -r requirements-ml.txt\n"
                     "(or use --no-ocr to test the marker finder only).")

    marker_rows, photo_rows = [], []
    for scene in [s.lower() for s in args.scenes]:
        photos = scene_files(raw, scene, "photo")
        if args.max_photos:
            photos = photos[: args.max_photos]
        if not photos:
            print(f"{scene}: no photos under {raw}. Did you run fetch_scenes.py pull?")
            continue
        valid = valid_numbers(raw, scene)
        print(f"{scene}: {len(photos)} photos, {len(valid)} numbers in the exhibit list {sorted(valid)}")
        (out / "overlays" / scene).mkdir(parents=True, exist_ok=True)

        for path in tqdm(photos, desc=scene):
            img = read_image(path)
            if img is None:
                print(f"  could not read {path.name}")
                continue
            boxes = find_markers(img, params)
            reads, labels = [], []
            for i, b in enumerate(boxes):
                row = {"scene_id": scene, "photo_id": path.stem, "marker_idx": i,
                       "x": b.x, "y": b.y, "w": b.w, "h": b.h, "method": "color_shape",
                       "det_score": round(b.score, 3), "raw_ocr": "", "ocr_conf": 0.0,
                       "marker_no": "", "status": "rejected"}
                if reader is not None:
                    from ffs.ocr import read_marker
                    res = read_marker(img, b, valid, reader, params)
                    row.update({k: ("" if v is None else v) for k, v in res.items()})
                marker_rows.append(row)
                if row["marker_no"] != "":
                    reads.append(str(row["marker_no"]))
                labels.append(f"{row['marker_no'] or '?'} ({row['raw_ocr'] or '-'})")
            photo_rows.append({"scene_id": scene, "photo_id": path.stem,
                               "relpath": path.relative_to(raw).as_posix(),
                               "n_boxes": len(boxes), "read_nos": ";".join(sorted(reads, key=int))})
            cv2.imwrite(str(out / "overlays" / scene / f"{path.stem}.jpg"),
                        draw_overlay(img, boxes, labels or None), [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not photo_rows:
        sys.exit("Nothing processed.")
    markers = pd.DataFrame(marker_rows, columns=[
        "scene_id", "photo_id", "marker_idx", "x", "y", "w", "h", "method", "det_score",
        "raw_ocr", "ocr_conf", "marker_no", "status"])
    photos_df = pd.DataFrame(photo_rows)
    markers.to_csv(out / "probe_markers.csv", index=False)
    photos_df.to_csv(out / "probe_photos.csv", index=False)

    truth_path = out / "truth.csv"
    if truth_path.exists() and not args.force:
        print(f"\n{truth_path} already exists and was NOT overwritten (your labels are safe).")
    else:
        sample = pick_sample(photos_df, args.sample_per_scene)[["scene_id", "photo_id", "relpath"]].copy()
        sample["n_markers_visible"] = ""
        sample["true_marker_nos"] = ""
        sample["notes"] = ""
        sample.to_csv(truth_path, index=False)
        print(f"\nWrote {truth_path} with {len(sample)} photos to label.")

    print(f"\nPhotos: {len(photos_df)} | photos with at least one box: {(photos_df.n_boxes > 0).sum()}"
          f" | boxes: {len(markers)}")
    print(f"Open the images in {out / 'overlays'} and compare them with the originals.")


if __name__ == "__main__":
    main()
