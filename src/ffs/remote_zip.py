"""Pull only the files we need out of a huge remote zip (SPEC 4.1).

The Zenodo zips are about 45 GB, almost all laser scans. `remotezip` reads the
zip's table of contents, then downloads only the entries we ask for, using HTTP
range requests. Nothing here depends on the exact folder layout: files are
recognised by the names of the folders they sit in (case-insensitive).
"""

from __future__ import annotations

import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

SCENE_DIR_RE = re.compile(r"^th_(\d{4})$", re.IGNORECASE)
SCENE_ANY_RE = re.compile(r"th_(\d{4})", re.IGNORECASE)
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
TABLE_EXTS = {".xlsx", ".xls", ".csv"}
BLOCKED_DIRS = {"pano", "scan", "ipc_fr", "__macosx"}  # R-A0-1


def _parts(name: str) -> list[str]:
    return [p for p in PurePosixPath(name.replace("\\", "/")).parts if p not in ("/", "")]


def scene_of(name: str) -> str | None:
    """Scene id such as 'th_0007', from a folder name (preferred) or the file name."""
    for part in _parts(name):
        if SCENE_DIR_RE.match(part):
            return part.lower()
    m = SCENE_ANY_RE.search(name)
    return f"th_{m.group(1)}" if m else None


def kind_of(name: str) -> str:
    """One of: photo, exhibit_list, description, blocked, other."""
    if name.endswith("/"):
        return "other"
    parts = _parts(name)
    if not parts:
        return "other"
    dirs = [p.lower() for p in parts[:-1]]
    stem = PurePosixPath(parts[-1]).stem.lower()
    ext = PurePosixPath(parts[-1]).suffix.lower()

    if any(d in BLOCKED_DIRS for d in dirs) or stem.endswith(("_fr", "_fra")):
        return "blocked"
    if "img" in dirs and ext in PHOTO_EXTS:
        return "photo"
    if ext in TABLE_EXTS and ("ipc_eng" in dirs or ("ipc" in dirs and stem.endswith("_eng"))):
        return "exhibit_list"
    if ext in TABLE_EXTS and "description" in dirs:
        return "description"
    return "other"


def wanted(name: str, scenes: set[str] | None) -> bool:
    """Do we download this entry? Description sheets are small and global."""
    kind = kind_of(name)
    if kind == "description":
        return True
    if kind not in ("photo", "exhibit_list"):
        return False
    scene = scene_of(name)
    return scene is not None and (scenes is None or scene in scenes)


def safe_dest(out_dir: Path, name: str) -> Path:
    """Where to write an entry. Refuses absolute paths and '..' (R-A0-2)."""
    parts = _parts(name)
    if not parts or any(p == ".." for p in parts) or re.match(r"^[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe path in zip: {name!r}")
    dest = Path(out_dir).joinpath(*parts)
    root = Path(out_dir).resolve()
    if root not in dest.resolve().parents and dest.resolve() != root:
        raise ValueError(f"Path escapes output folder: {name!r}")
    return dest


def open_remote(url: str):
    """Open a remote zip. Needs `pip install remotezip`."""
    from remotezip import RemoteZip

    return RemoteZip(url)


def summarize(infos) -> dict:
    """Layout summary of a zip's entries (objects with .filename and .file_size)."""
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    top_dirs: Counter = Counter()
    scenes: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[str]] = defaultdict(list)

    for info in infos:
        name = info.filename
        kind = kind_of(name)
        by_kind[kind][0] += 1
        by_kind[kind][1] += info.file_size
        parts = _parts(name)
        if parts:
            top_dirs[parts[0]] += 1
        if len(samples[kind]) < 8 and not name.endswith("/"):
            samples[kind].append(name)
        sc = scene_of(name)
        if sc and kind in ("photo", "exhibit_list"):
            scenes[sc][kind] += 1

    return {
        "n_entries": sum(v[0] for v in by_kind.values()),
        "by_kind": {k: tuple(v) for k, v in by_kind.items()},
        "top_dirs": dict(top_dirs.most_common(12)),
        "scenes": {s: dict(c) for s, c in sorted(scenes.items())},
        "samples": dict(samples),
    }


def format_summary(summary: dict) -> str:
    gb = 1024**3
    lines = [f"Entries in zip: {summary['n_entries']}", "", "By kind (count, size):"]
    for kind, (n, size) in sorted(summary["by_kind"].items()):
        lines.append(f"  {kind:<13} {n:>8}  {size / gb:8.2f} GB")
    lines += ["", "Top-level names (count):"]
    lines += [f"  {k}: {v}" for k, v in summary["top_dirs"].items()]
    lines += ["", "Example paths per kind:"]
    for kind, names in sorted(summary["samples"].items()):
        lines.append(f"  [{kind}]")
        lines += [f"    {n}" for n in names]
    lines += ["", "Scenes (photos, exhibit lists):"]
    for sc, c in summary["scenes"].items():
        lines.append(f"  {sc}: {c.get('photo', 0)} photos, {c.get('exhibit_list', 0)} exhibit list(s)")
    return "\n".join(lines)


def extract_wanted(zf, infos, out_dir, scenes: set[str] | None = None,
                   dry_run: bool = False, retries: int = 3) -> dict:
    """Download the wanted entries. Skips files already present with the same size (R-A0-3)."""
    from tqdm import tqdm

    todo = [i for i in infos if wanted(i.filename, scenes)]
    stats = {"planned": len(todo), "planned_bytes": sum(i.file_size for i in todo),
             "downloaded": 0, "skipped": 0}
    if dry_run:
        return stats

    for info in tqdm(todo, desc="downloading", unit="file"):
        dest = safe_dest(Path(out_dir), info.filename)
        if dest.exists() and dest.stat().st_size == info.file_size:
            stats["skipped"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")  # never leave a half file that looks complete
        for attempt in range(1, retries + 1):
            try:
                with zf.open(info) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                tmp.replace(dest)
                stats["downloaded"] += 1
                break
            except Exception:
                if attempt == retries:
                    raise
    return stats
