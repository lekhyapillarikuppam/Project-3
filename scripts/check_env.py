"""Check that the laptop is ready. Run:  python scripts/check_env.py

Core packages are required now. ML packages are only needed from Phase 1.
"""

import importlib
import shutil
import sys
from pathlib import Path

CORE = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
    ("cv2", "opencv-python"), ("PIL", "pillow"), ("tqdm", "tqdm"),
    ("requests", "requests"), ("remotezip", "remotezip"), ("yaml", "pyyaml"),
    ("openpyxl", "openpyxl"), ("pytest", "pytest"),
]
ML = [
    ("torch", "torch"), ("transformers", "transformers"),
    ("open_clip", "open_clip_torch"), ("easyocr", "easyocr"),
]
DIRS = ["src/ffs", "tests", "scripts", "docs", "data", "data_meta",
        "results", "notebooks", "app"]


def check(modules):
    missing = []
    for mod, pip_name in modules:
        try:
            m = importlib.import_module(mod)
            print(f"  OK       {pip_name:<16} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  MISSING  {pip_name:<16} ({type(e).__name__})")
            missing.append(pip_name)
    return missing


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    problems = 0

    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    print(f"Python {v.major}.{v.minor}.{v.micro}: {'OK' if ok else 'TOO OLD (need 3.10+)'}")
    problems += not ok

    print(f"git: {'OK' if shutil.which('git') else 'MISSING'}")
    problems += not shutil.which("git")

    print("\nRepo layout:")
    for d in DIRS:
        exists = (root / d).is_dir()
        print(f"  {'OK     ' if exists else 'MISSING'}  {d}/")
        problems += not exists

    print("\nCore packages (required now):")
    problems += len(check(CORE))

    print("\nML packages (needed from Phase 1; MISSING is fine today):")
    check(ML)

    print("\nRESULT:", "all good" if problems == 0 else f"{problems} problem(s) to fix")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
