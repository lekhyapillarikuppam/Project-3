"""Project-wide constants. Values here must match SPEC.md (sections 2 and 3)."""

from pathlib import Path

# Paths ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"            # raw photos and zips; gitignored
RAW_DIR = DATA_DIR / "raw"               # downloads; never edit by hand
META_DIR = REPO_ROOT / "data_meta"       # small CSVs; committed
RESULTS_DIR = REPO_ROOT / "results"
APP_DIR = REPO_ROOT / "app"
FEASIBILITY_DIR = RESULTS_DIR / "feasibility"

# Reproducibility -----------------------------------------------------------
SEED = 42

# Split (SPEC D-004) ----------------------------------------------------------
DEV_FRACTION = 0.4
SPLITS = ("dev", "test")

# Category inclusion rule (SPEC D-003) ----------------------------------------
MIN_POSITIVE_PHOTOS = 15
MIN_POSITIVE_SCENES = 6

# Categories (SPEC 3.1). Final 6-8 are chosen in Phase 2. ---------------------
CANDIDATE_CATEGORIES = (
    "blade",
    "firearm",
    "ammunition",
    "bottle_glass",
    "phone_electronics",
    "paper_document",
    "footwear",
    "red_brown_stain",
)
OTHER = "other"

# Allowed values (SPEC 2) -------------------------------------------------------
PHOTO_TYPES = ("overview", "mid", "close", "unknown")
CATEGORY_SOURCES = ("rule", "llm", "manual")
MARKER_STATUSES = ("accepted", "snapped", "rejected")
MARKER_METHODS = ("color_shape", "owlv2")

# Evaluation (SPEC 6) -----------------------------------------------------------
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# Dataset (SPEC 4.1) -------------------------------------------------------------
# Part 1 = scenes th_0001-0020. Other parts live in other Zenodo records; pass --url for them.
ZENODO_ZIP_URLS = {
    1: "https://zenodo.org/records/17151982/files/th_0001-0020.zip?download=1",
}

# Feasibility gate pass marks (SPEC 8.2) ---------------------------------------
FEAS_MIN_FINDER_HIT = 0.70
FEAS_MIN_READ_ACCURACY = 0.60
