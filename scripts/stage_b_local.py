"""Stage B on the laptop CPU (SPEC 5, R-B-10): CLIP whole-photo scores and OWLv2 boxes on tile 0 only.

  python scripts/stage_b_local.py --split test                 # full run; run the same command again to resume
  python scripts/stage_b_local.py --split test --limit 5       # smoke test on the first 5 photos

Reads the ORIGINAL photos (data/raw, via data_meta/photos.csv), not the 2400 px upload copies.
Writes results/stage_b/<split>_outputs/clip_scores_<split>.csv, owlv2_raw_<split>.csv and run_info_<split>.json
in the same format as the Kaggle run (kaggle/stage_b_run.py). Prompts, queries, models, cut-off and top-k are
imported from that file, never copied, so the frozen protocol has one source.
This script never reads silver, gold or exhibit lists. Test photos are scored only after the tag protocol-v1.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ffs import config  # noqa: E402
from package_stage_b import TAG, protocol_frozen  # noqa: E402

# Import the Kaggle file as a module (it lives outside any package).
_spec = importlib.util.spec_from_file_location("stage_b_run", ROOT / "kaggle" / "stage_b_run.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)

CLIP_HEADER = ["scene_id", "photo_id", "category", "prompt", "score"]
OWL_HEADER = ["scene_id", "photo_id", "category", "prompt", "score", "x", "y", "w", "h", "tile"]
WRITE_EVERY = 10      # photos per write, so a stopped run loses at most this many
PROGRESS_EVERY = 25
N_THREADS = 8


# ---- One photo -> output rows (models are passed in, so tests can use a fake) ----------------
def photo_rows(img, scene_id: str, photo_id: str, clip_fn, owl_fn):
    """Return (clip_rows, owl_rows, clip_seconds, owl_seconds) for one photo in original pixels.

    clip_fn(img) -> logits, one per prompt of flat_queries(CLIP_PROMPTS).
    owl_fn(img) -> (scores [patches, queries], boxes [patches, 4]) for the full photo (tile 0).
    """
    clip_q, owl_q = sb.flat_queries(sb.CLIP_PROMPTS), sb.flat_queries(sb.OWL_QUERIES)
    t0 = time.time()
    logits = clip_fn(img)
    t1 = time.time()
    clip_rows = [[scene_id, photo_id, cat, prompt, round(float(sc), 4)] for (cat, prompt), sc in zip(clip_q, logits)]

    W, H = img.size
    scores, boxes = owl_fn(img)
    t2 = time.time()
    # Tile 0 is the full photo at original size, so no offset and scale 1.
    dec = sb.decode_boxes(scores, boxes, side=max(W, H), off_x=0, off_y=0, max_x=W, max_y=H, scale=1.0)
    owl_rows = []
    for qi, found in dec.items():
        cat, prompt = owl_q[qi]
        for s, x, y, bw, bh in found:
            owl_rows.append([scene_id, photo_id, cat, prompt, round(s, 4),
                             round(x, 1), round(y, 1), round(bw, 1), round(bh, 1), 0])
    return clip_rows, owl_rows, t1 - t0, t2 - t1


# ---- Resume ---------------------------------------------------------------------------------
def read_done(out: Path, name: str) -> set[tuple[str, str]]:
    """Photos already finished. A photo counts only if all its CLIP rows are there; any half-written
    photo is removed from both files so it is redone cleanly."""
    f_clip, f_owl = out / f"clip_scores_{name}.csv", out / f"owlv2_raw_{name}.csv"
    if not f_clip.exists() or not f_owl.exists():
        return set()
    n_prompts = len(sb.flat_queries(sb.CLIP_PROMPTS))
    clip = list(csv.reader(open(f_clip, encoding="utf-8")))[1:]
    owl = list(csv.reader(open(f_owl, encoding="utf-8")))[1:]
    counts: dict[tuple[str, str], int] = {}
    for r in clip:
        counts[(r[0], r[1])] = counts.get((r[0], r[1]), 0) + 1
    done = {k for k, n in counts.items() if n == n_prompts}
    for path, header, rows in [(f_clip, CLIP_HEADER, clip), (f_owl, OWL_HEADER, owl)]:
        kept = [r for r in rows if (r[0], r[1]) in done]
        if len(kept) != len(rows):
            write_csv(path, header, kept, mode="w")
    return done


def write_csv(path: Path, header, rows, mode: str = "a"):
    new = mode == "w" or not path.exists()
    with open(path, "w" if new else "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


# ---- The run loop ---------------------------------------------------------------------------
def run(photos: list[dict], out: Path, name: str, clip_fn, owl_fn, load=sb.load_image, info_extra=None):
    """Score every photo not already in the output files. photos: dicts with scene_id, photo_id, path."""
    out.mkdir(parents=True, exist_ok=True)
    f_clip, f_owl = out / f"clip_scores_{name}.csv", out / f"owlv2_raw_{name}.csv"
    f_info = out / f"run_info_{name}.json"
    done = read_done(out, name)
    if not done:                                   # fresh start: empty files with headers
        write_csv(f_clip, CLIP_HEADER, [], mode="w")
        write_csv(f_owl, OWL_HEADER, [], mode="w")
    prev_seconds = 0.0
    if done and f_info.exists():
        prev_seconds = json.loads(f_info.read_text(encoding="utf-8")).get("seconds", 0)
    todo = [p for p in photos if (p["scene_id"], p["photo_id"]) not in done]
    print(f"{len(done)} photos already done, {len(todo)} to go", flush=True)

    info = {"name": name, "n_photos": len(photos), "clip_model": sb.CLIP_MODEL, "owl_model": sb.OWL_MODEL,
            "clip_prompts": sb.CLIP_PROMPTS, "owl_queries": sb.OWL_QUERIES, "tiles": "tile0 only",
            "raw_cutoff": sb.RAW_CUTOFF, "top_k": sb.TOP_K, **(info_extra or {})}

    def save(n_done: int, seconds: float):
        info.update({"n_photos_done": n_done, "complete": n_done == len(photos), "seconds": round(seconds)})
        f_info.write_text(json.dumps(info, indent=2), encoding="utf-8")

    t0 = time.time()
    buf_clip, buf_owl, t_clip, t_owl = [], [], 0.0, 0.0
    for n, p in enumerate(todo, 1):
        img = load(p["path"])
        c, o, tc, to = photo_rows(img, p["scene_id"], p["photo_id"], clip_fn, owl_fn)
        buf_clip += c
        buf_owl += o
        t_clip, t_owl = t_clip + tc, t_owl + to
        if n % WRITE_EVERY == 0 or n == len(todo):
            write_csv(f_owl, OWL_HEADER, buf_owl)  # OWLv2 first: a photo counts as done only once CLIP is written
            write_csv(f_clip, CLIP_HEADER, buf_clip)
            buf_clip, buf_owl = [], []
            save(len(done) + n, prev_seconds + time.time() - t0)
        if n % PROGRESS_EVERY == 0 or n == len(todo):
            per = (time.time() - t0) / n
            eta = per * (len(todo) - n)
            print(f"{len(done) + n}/{len(photos)} photos | {per:.1f} s/photo (CLIP {t_clip / n:.1f}, "
                  f"OWLv2 {t_owl / n:.1f}) | ETA {eta / 60:.0f} min", flush=True)
    if not todo:
        save(len(done), prev_seconds)
    return info


# ---- Real models on CPU ---------------------------------------------------------------------
def load_models():
    """Return (clip_fn, owl_fn, versions) using the frozen models on CPU."""
    import torch
    import transformers
    from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor

    torch.set_num_threads(N_THREADS)
    clip = CLIPModel.from_pretrained(sb.CLIP_MODEL).eval()
    clip_proc = CLIPProcessor.from_pretrained(sb.CLIP_MODEL)
    owl = Owlv2ForObjectDetection.from_pretrained(sb.OWL_MODEL).eval()
    owl_proc = Owlv2Processor.from_pretrained(sb.OWL_MODEL)
    clip_texts = [p for _, p in sb.flat_queries(sb.CLIP_PROMPTS)]
    owl_texts = [[p for _, p in sb.flat_queries(sb.OWL_QUERIES)]]

    def clip_fn(img):                              # same call as the Kaggle run: whole photo padded to a square
        with torch.inference_mode():
            inp = clip_proc(text=clip_texts, images=sb.pad_square(img), return_tensors="pt", padding=True)
            return clip(**inp).logits_per_image[0].float().numpy()

    def owl_fn(img):
        with torch.inference_mode():
            inp = owl_proc(text=owl_texts, images=img, return_tensors="pt")
            o = owl(**inp)
        return o.logits[0].sigmoid().float().numpy(), o.pred_boxes[0].float().numpy()

    import PIL
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__,
                "numpy": sb.np.__version__, "python": sys.version.split()[0], "device": "cpu",
                "torch_threads": torch.get_num_threads()}
    return clip_fn, owl_fn, versions


def split_photos(split: str) -> list[dict]:
    """Photos of a split in the same order as the Kaggle manifest (scene, photo), with ORIGINAL file paths."""
    import pandas as pd

    photos = pd.read_csv(config.META_DIR / "photos.csv")
    sp = pd.read_csv(config.META_DIR / "scene_split.csv")
    scenes = set(sp.loc[sp["split"] == split, "scene_id"])
    p = photos[photos["scene_id"].isin(scenes)].sort_values(["scene_id", "photo_id"])
    return [{"scene_id": r.scene_id, "photo_id": r.photo_id, "path": config.RAW_DIR / r.relpath}
            for r in p.itertuples()]


def git_describe() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"git_tag": TAG if protocol_frozen(git("tag", "-l", TAG).split()) else None,
            "git_tag_commit": git("rev-list", "-n1", TAG) or None, "git_head": git("rev-parse", "HEAD")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos (smoke test)")
    ap.add_argument("--out", default=None, help="default results/stage_b/<split>_outputs")
    args = ap.parse_args()

    gitinfo = git_describe()
    if args.split == "test" and gitinfo["git_tag"] is None:
        sys.exit(f"Refusing: the git tag '{TAG}' does not exist. Test photos are not scored before the "
                 f"protocol is frozen (SPEC 6, D-017).")
    if args.split == "dev" and not args.out:      # results/stage_b/dev_outputs holds the Kaggle dev run
        sys.exit("For dev, pass --out to a new folder so the Kaggle dev outputs are not touched.")
    out = Path(args.out) if args.out else config.RESULTS_DIR / "stage_b" / f"{args.split}_outputs"
    photos = split_photos(args.split)[: args.limit]
    clip_fn, owl_fn, versions = load_models()
    run(photos, out, args.split, clip_fn, owl_fn, info_extra={**versions, **gitinfo})
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
