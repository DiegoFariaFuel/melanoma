#!/usr/bin/env python3
"""
Small helper to verify repo folders and count image files.
"""
from pathlib import Path
import os

EXTS = {'.jpg', '.jpeg', '.png'}


def summarize_path(p: Path):
    out = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out

    if p.is_dir():
        total = 0
        images = 0
        samples = []
        for root, _, files in os.walk(p):
            for f in files:
                total += 1
                if Path(f).suffix.lower() in EXTS:
                    images += 1
                if len(samples) < 10:
                    samples.append(os.path.relpath(os.path.join(root, f), start=p))
        out.update({"type": "dir", "total_files": total, "image_files": images, "samples": samples[:10]})
        return out

    # file
    try:
        size = p.stat().st_size
    except Exception:
        size = None

    lines = None
    if p.suffix.lower() in ('.csv', '.txt'):
        try:
            with p.open('rb') as fh:
                lines = sum(1 for _ in fh)
        except Exception:
            lines = None

    out.update({"type": "file", "size_bytes": size, "lines": lines})
    return out


def main():
    root = Path(__file__).resolve().parents[1]

    checks = {
        "repo_root": root,
        "predata": root / "predata.py",
        "default_images": root / "images",
        "archive": root / "archive",
        "archive_images": root / "archive" / "images",
        "archive_masks": root / "archive" / "masks",
        "archive_labels_csv": root / "archive" / "labels.csv",
        "archive_groundtruth_csv": root / "archive" / "GroundTruth.csv",
        "saida_default": root / "data" / "processado",
    }

    for name, path in checks.items():
        s = summarize_path(path)
        if not s.get("exists"):
            print(f"{name}: MISSING -> {s['path']}")
            continue

        if s.get("type") == "dir":
            print(f"{name}: dir, total_files={s['total_files']}, image_files={s['image_files']}, sample={s['samples'][:5]}")
        else:
            print(f"{name}: file, size={s.get('size_bytes')}, lines={s.get('lines')}")


if __name__ == '__main__':
    main()
