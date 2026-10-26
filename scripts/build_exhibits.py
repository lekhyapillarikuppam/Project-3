"""Write data_meta/exhibits.csv from the downloaded exhibit lists (SPEC R-P-1).

  python scripts/build_exhibits.py

Download the lists first:  python scripts/fetch_scenes.py pull --scenes th_0001 ... th_0020
Rows with an empty marker number are dropped and counted. Repeated numbers are kept.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, remote_zip as rz  # noqa: E402
from ffs.dataprep import clean_exhibits  # noqa: E402
from ffs.exhibits import read_exhibit_rows  # noqa: E402
from ffs.schemas import validate  # noqa: E402


def exhibit_list_files(raw: Path) -> dict[str, Path]:
    """scene_id -> its English exhibit-list file."""
    out = {}
    for f in sorted(raw.rglob("*")):
        if f.is_file():
            rel = f.relative_to(raw).as_posix()
            if rz.kind_of(rel) == "exhibit_list":
                out[rz.scene_of(rel)] = f
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(config.RAW_DIR))
    ap.add_argument("--out", default=str(config.META_DIR / "exhibits.csv"))
    args = ap.parse_args()

    frames, n_dropped = [], 0
    for scene, path in exhibit_list_files(Path(args.raw)).items():
        clean, dropped = clean_exhibits(read_exhibit_rows(path).assign(scene_id=scene))
        n_dropped += dropped
        if dropped:
            print(f"  {scene}: {dropped} row(s) with an empty marker number dropped")
        frames.append(clean)
    if not frames:
        sys.exit("No exhibit lists found. Run fetch_scenes.py pull first.")

    exhibits = pd.concat(frames, ignore_index=True)[["scene_id", "marker_no", "description"]]
    validate(exhibits, "exhibits")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    exhibits.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(exhibits)} exhibits in {exhibits.scene_id.nunique()} scenes. "
          f"Rows with an empty marker number: {n_dropped}.")


if __name__ == "__main__":
    main()
