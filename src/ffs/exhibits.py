"""Read an exhibit-list spreadsheet into (marker_no, description) rows (SPEC 4.2).

The real column layout is not known yet, so the reader guesses:
  marker column      = the column whose cells are mostly whole numbers, mostly unique
  description column = the other column with the most text
and `scripts/inspect_exhibits.py` lets you confirm the guess by eye.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INT_RE = re.compile(r"^\s*(\d+)(\.0+)?\s*$")


def _as_int(value) -> int | None:
    m = INT_RE.match(str(value))
    return int(m.group(1)) if m else None


def _read_grid(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    """The sheet as a raw grid of strings, with no header handling."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        try:
            grid = pd.read_csv(path, header=None, dtype=object, keep_default_na=False)
        except UnicodeDecodeError:
            grid = pd.read_csv(path, header=None, dtype=object, keep_default_na=False,
                               encoding="latin-1")
    else:
        sheets = pd.read_excel(path, sheet_name=sheet if sheet is not None else None,
                               header=None, dtype=object)
        if isinstance(sheets, dict):  # several sheets: take the one with the most filled cells
            grid = max(sheets.values(), key=lambda d: int(d.notna().sum().sum()))
        else:
            grid = sheets
    return grid.fillna("").astype(str)


def guess_columns(grid: pd.DataFrame) -> tuple[int, int]:
    """Return (marker_col, description_col) as positional column indexes."""
    best, marker_col = -1.0, None
    for i in range(grid.shape[1]):
        ints = [n for n in (_as_int(v) for v in grid.iloc[:, i] if str(v).strip()) if n is not None]
        if not ints:
            continue
        score = len(ints) * (len(set(ints)) / len(ints))  # many numbers, mostly unique
        if score > best:
            best, marker_col = score, i
    if marker_col is None:
        raise ValueError("No column of whole numbers found: cannot locate marker numbers.")

    best, desc_col = -1, None
    for i in range(grid.shape[1]):
        if i == marker_col:
            continue
        texts = [str(v).strip() for v in grid.iloc[:, i]
                 if str(v).strip() and _as_int(v) is None]
        total = sum(len(t) for t in texts)
        if total > best:
            best, desc_col = total, i
    if desc_col is None or best <= 0:
        raise ValueError("No text column found: cannot locate descriptions.")
    return marker_col, desc_col


def read_exhibit_file(path, marker_col: int | None = None, desc_col: int | None = None,
                      sheet: str | int | None = None) -> pd.DataFrame:
    """Columns: marker_no (int), description (str). Rows without a number are skipped (R-A1-3)."""
    grid = _read_grid(Path(path), sheet)
    if marker_col is None or desc_col is None:
        g_marker, g_desc = guess_columns(grid)
        marker_col = g_marker if marker_col is None else marker_col
        desc_col = g_desc if desc_col is None else desc_col

    rows = []
    for _, row in grid.iterrows():
        n = _as_int(row.iloc[marker_col])
        desc = str(row.iloc[desc_col]).strip()
        if n is not None and desc:
            rows.append({"marker_no": n, "description": desc})
    return pd.DataFrame(rows, columns=["marker_no", "description"])
