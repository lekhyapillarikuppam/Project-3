"""Show exhibit lists the way the reader sees them, so you can confirm the columns (SPEC 4.2, R-A1-4).

  python scripts/inspect_exhibits.py --scene th_0003
  python scripts/inspect_exhibits.py path/to/list.xlsx
  python scripts/inspect_exhibits.py path/to/list.xlsx --marker-col 0 --desc-col 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ffs import config, remote_zip as rz  # noqa: E402
from ffs.exhibits import _read_grid, guess_columns, read_exhibit_file  # noqa: E402


def show(path: Path, marker_col, desc_col):
    print("=" * 78 + f"\n{path}\n" + "=" * 78)
    grid = _read_grid(path)
    print(f"Raw grid ({grid.shape[0]} rows x {grid.shape[1]} columns), first 12 rows "
          "(columns are numbered from 0):")
    with pd.option_context("display.max_colwidth", 40, "display.width", 200):
        print(grid.head(12).to_string())
    try:
        g_m, g_d = guess_columns(grid)
        print(f"\nGuessed columns: marker number = column {g_m}, description = column {g_d}")
        df = read_exhibit_file(path, marker_col, desc_col)
    except ValueError as e:
        print(f"\nCould not read: {e}\nTry --marker-col N --desc-col M.")
        return
    print(f"\nReader result: {len(df)} exhibits")
    print(df.to_string(index=False))
    dup = df["marker_no"][df["marker_no"].duplicated()].tolist()
    print(f"\nMarker numbers: {sorted(df['marker_no'].tolist())}")
    if dup:
        print(f"NOTE: repeated marker numbers {sorted(set(dup))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="spreadsheet files to inspect")
    ap.add_argument("--scene", help="inspect the exhibit list(s) of a downloaded scene, e.g. th_0003")
    ap.add_argument("--raw", default=str(config.RAW_DIR))
    ap.add_argument("--marker-col", type=int)
    ap.add_argument("--desc-col", type=int)
    args = ap.parse_args()

    files = [Path(p) for p in args.paths]
    if args.scene:
        raw = Path(args.raw)
        files += [f for f in raw.rglob("*") if f.is_file()
                  and rz.kind_of(f.relative_to(raw).as_posix()) == "exhibit_list"
                  and rz.scene_of(f.relative_to(raw).as_posix()) == args.scene.lower()]
    if not files:
        sys.exit("No files to inspect. Give a path, or --scene after running fetch_scenes.py pull.")
    for f in files:
        show(f, args.marker_col, args.desc_col)


if __name__ == "__main__":
    main()
