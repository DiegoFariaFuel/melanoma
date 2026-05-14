#!/usr/bin/env python3
r"""ONNX inference helper using onnxruntime.

Usage examples:
    # single image
    & .venv\Scripts\python.exe scripts/onnx_infer.py --onnx-path models/best_model.onnx --image-path data/split/validacao/nv/ISIC_XXXXX.jpg

    # directory (will save CSV)
    & .venv\Scripts\python.exe scripts/onnx_infer.py --onnx-path models/best_model.onnx --image-dir data/split/validacao --output-csv results/predictions_onnx.csv
"""

import argparse
import json
from pathlib import Path
import os
import sys

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime") from e


def parse_args():
    p = argparse.ArgumentParser(description="Run ONNX model inference against images using onnxruntime")
    p.add_argument('--onnx-path', type=Path, default=Path('models/best_model.onnx'))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image-path', type=Path, help='Single image file to run')
    g.add_argument('--image-dir', type=Path, help='Directory with images (recursively scanned)')
    p.add_argument('--val-dir', type=Path, default=Path('data/split/validacao'), help='Validation folder to infer class order from')
    p.add_argument('--classes-file', type=Path, help='JSON file with class name list (overrides --val-dir)')
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--use-cuda', action='store_true', help='Try to run on CUDAExecutionProvider if available')
    p.add_argument('--input-size', type=int, default=224, help='Input image size (height/width)')
    p.add_argument('--output-csv', type=Path, default=Path('results/predictions_onnx.csv'))
    return p.parse_args()


def get_class_names(val_dir: Path = None, classes_file: Path = None):
    if classes_file and classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            names = json.load(f)
            if not isinstance(names, list):
                raise RuntimeError('classes-file must contain a JSON list of class names')
            return names
    if val_dir is None or not Path(val_dir).exists():
        raise RuntimeError('Cannot determine class names: provide --val-dir or --classes-file')
    # ImageFolder/torchvision uses sorted directory names as class order
    dirs = [p for p in sorted(Path(val_dir).iterdir()) if p.is_dir()]
    return [d.name for d in dirs]


def preprocess_image(path: Path, size=(224, 224)) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)  # CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    arr = np.ascontiguousarray(arr[np.newaxis, ...].astype(np.float32))
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def predict(session: ort.InferenceSession, input_name: str, img_array: np.ndarray, topk: int = 3):
    outputs = session.run(None, {input_name: img_array})
    logits = outputs[0]
    probs = softmax(logits)
    idx = np.argsort(probs, axis=1)[:, ::-1][:, :topk]
    return probs, idx


def find_images(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = [p for p in folder.rglob('*') if p.suffix.lower() in exts]
    return sorted(files)


def main():
    args = parse_args()
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx_path}")

    class_names = get_class_names(args.val_dir, args.classes_file)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.use_cuda else ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(args.onnx_path), providers=providers)
    except Exception:
        # fallback to default session creation
        session = ort.InferenceSession(str(args.onnx_path))

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = input_meta.shape
    try:
        ih = int(shape[2])
        iw = int(shape[3])
    except Exception:
        ih = iw = args.input_size

    if args.image_path:
        img = preprocess_image(args.image_path, size=(ih, iw))
        probs, idx = predict(session, input_name, img, args.topk)
        for rank, i in enumerate(idx[0], start=1):
            cls = class_names[int(i)] if int(i) < len(class_names) else str(i)
            p = float(probs[0, int(i)])
            print(f"Top{rank}: {cls} ({p:.4f})")
        return

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    images = find_images(args.image_dir)
    if not images:
        print(f"No images found in {args.image_dir}")
        return

    rows = []
    for p in images:
        try:
            img = preprocess_image(p, size=(ih, iw))
            probs, idx = predict(session, input_name, img, args.topk)
            top_idx = idx[0]
            top_classes = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in top_idx]
            top_probs = [float(probs[0, int(i)]) for i in top_idx]
            rows.append({'image': str(p), 'top1': top_classes[0], 'top1_prob': top_probs[0], 'topk': ';'.join(top_classes)})
        except Exception as e:
            rows.append({'image': str(p), 'top1': '', 'top1_prob': 0.0, 'topk': '',})

    # write CSV
    import csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'top1', 'top1_prob', 'topk'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote predictions for {len(rows)} images to {out_csv}")


if __name__ == '__main__':
    main()
