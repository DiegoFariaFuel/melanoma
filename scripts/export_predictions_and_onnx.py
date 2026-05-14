import os
import argparse
import json
import numpy as np
import pandas as pd
import sys
import torch
from torchvision import datasets
import seaborn as sns

# ensure project root is on sys.path so we can import train_melanoma
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from train_melanoma_old import build_model, get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/best_checkpoint.pth')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--val-dir', type=str, default='data/split/validacao')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--onnx-path', type=str, default='models/best_model.onnx')
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cpu') if args.device == 'cpu' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # dataset
    _, val_transform = get_transforms()
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = val_dataset.classes
    num_classes = len(class_names)

    # model
    model = build_model(num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    # remove module. prefix if present
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state)
    model.to(device)
    model.eval()

    # inference
    file_paths = [p for p, _ in val_dataset.samples]
    N = len(val_dataset)
    all_probs = np.zeros((N, num_classes), dtype=float)
    all_preds = np.zeros(N, dtype=int)

    idx = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            bs = probs.shape[0]
            all_probs[idx:idx+bs] = probs
            all_preds[idx:idx+bs] = np.argmax(probs, axis=1)
            idx += bs

    # save predictions
    rows = []
    for i, path in enumerate(file_paths):
        true_label = val_dataset.targets[i]
        pred_label = int(all_preds[i])
        pred_conf = float(all_probs[i, pred_label])
        row = {
            'filepath': path,
            'true_label': int(true_label),
            'true_name': class_names[true_label],
            'pred_label': pred_label,
            'pred_name': class_names[pred_label],
            'pred_confidence': pred_conf
        }
        for c_idx, cname in enumerate(class_names):
            row[f'prob_{cname}'] = float(all_probs[i, c_idx])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    print('Saved predictions to', os.path.join(results_dir, 'predictions.csv'))

    # classification report
    report_dict = classification_report([int(x) for x in val_dataset.targets], all_preds, target_names=class_names, zero_division=0, output_dict=True)
    with open(os.path.join(results_dir, 'report_eval.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)
    with open(os.path.join(results_dir, 'report_eval.txt'), 'w') as f:
        f.write(classification_report([int(x) for x in val_dataset.targets], all_preds, target_names=class_names, zero_division=0))
    print('Saved classification report to', results_dir)

    # PR curves and AP
    y_true_bin = label_binarize(val_dataset.targets, classes=list(range(num_classes)))
    ap_scores = {}
    plt.figure(figsize=(10, 8))
    for i, cname in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], all_probs[:, i])
        ap_scores[cname] = float(ap)
        plt.plot(recall, precision, label=f"{cname} (AP={ap:.3f})")
        # also save per-class
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR curve - {cname} (AP={ap:.3f})')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'pr_curve_{cname}.png'))
        plt.close()

    plt.figure(figsize=(10, 8))
    for i, cname in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        ap = ap_scores[cname]
        plt.plot(recall, precision, label=f"{cname} (AP={ap:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'pr_curves.png'))
    plt.close()
    with open(os.path.join(results_dir, 'pr_aps.json'), 'w') as f:
        json.dump(ap_scores, f, indent=2)
    print('Saved PR curves and AP scores to', results_dir)

    # confusion matrix
    cm = confusion_matrix([int(x) for x in val_dataset.targets], all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_eval.png'))
    plt.close()
    print('Saved confusion matrix to', os.path.join(results_dir, 'confusion_matrix_eval.png'))

    # export ONNX
    if args.export_onnx:
        onnx_path = args.onnx_path
        model_cpu = model.to('cpu')
        model_cpu.eval()
        dummy = torch.randn(1, 3, 224, 224)
        try:
            torch.onnx.export(model_cpu, dummy, onnx_path, export_params=True, opset_version=11,
                              do_constant_folding=True, input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            print('Exported ONNX model to', onnx_path)
        except Exception as e:
            print('ONNX export failed:', e)


if __name__ == '__main__':
    main()
