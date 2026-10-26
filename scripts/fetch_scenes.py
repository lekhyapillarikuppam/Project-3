"""Fetch only the photos and English exhibit lists for chosen scenes (SPEC 4.1).

  python scripts/fetch_scenes.py list
  python scripts/fetch_scenes.py pull --scenes th_0003 th_0007 --dry-run
  python scripts/fetch_scenes.py pull --scenes th_0003 th_0007

`list` reads only the zip's table of contents (a few MB), prints the layout, and
saves it to data/raw/. Panoramas and laser scans are never downloaded.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffs import config, remote_zip as rz  # noqa: E402


def get_url(args) -> str:
    if args.url:
        return args.url
    if args.part not in config.ZENODO_ZIP_URLS:
        sys.exit(f"No URL known for part {args.part}. Pass --url with the zip's download link.")
    return config.ZENODO_ZIP_URLS[args.part]


def open_zip(url: str):
    print(f"Opening remote zip (reads only its table of contents)...\n  {url}")
    try:
        return rz.open_remote(url)
    except ImportError:
        sys.exit("Package 'remotezip' is missing. Run: pip install -r requirements.txt")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not open the remote zip: {type(e).__name__}: {e}\n"
                 "Check your internet connection and try again.")


def cmd_list(args):
    zf = open_zip(get_url(args))
    infos = zf.infolist()
    summary = rz.summarize(infos)
    text = rz.format_summary(summary)
    print("\n" + text)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RAW_DIR / f"zip_summary_part{args.part}.txt"
    out.write_text(text, encoding="utf-8")
    names = config.RAW_DIR / f"zip_names_part{args.part}.txt"
    names.write_text("\n".join(i.filename for i in infos
                               if rz.kind_of(i.filename) in ("photo", "exhibit_list", "description")),
                     encoding="utf-8")
    print(f"\nSaved: {out}\n       {names}")
    print("Next: paste the summary above into the chat so the spec can record the real layout.")


def cmd_pull(args):
    scenes = {s.lower() for s in args.scenes}
    bad = [s for s in scenes if not re.fullmatch(r"th_\d{4}", s)]
    if bad:
        sys.exit(f"Scene ids look like th_0007. Not understood: {bad}")

    zf = open_zip(get_url(args))
    infos = zf.infolist()
    present = set(rz.summarize(infos)["scenes"])
    missing = sorted(scenes - present)
    if missing:
        print(f"WARNING: no photos or lists found for {missing} in this zip.")

    plan = rz.extract_wanted(zf, infos, config.RAW_DIR, scenes, dry_run=True)
    print(f"\nWill download {plan['planned']} files, about {plan['planned_bytes'] / 1e6:.0f} MB, "
          f"into {config.RAW_DIR}")
    if args.dry_run:
        print("Dry run: nothing downloaded.")
        return
    stats = rz.extract_wanted(zf, infos, config.RAW_DIR, scenes)
    print(f"Done. downloaded={stats['downloaded']} skipped(already there)={stats['skipped']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, default=1, help="dataset part number (default 1 = th_0001-0020)")
    ap.add_argument("--url", help="override the zip download URL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the layout of the remote zip")
    p = sub.add_parser("pull", help="download photos and English lists for some scenes")
    p.add_argument("--scenes", nargs="+", required=True, help="e.g. th_0003 th_0007")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    args = ap.parse_args()
    {"list": cmd_list, "pull": cmd_pull}[args.cmd](args)


if __name__ == "__main__":
    main()
