#!/usr/bin/env python3
"""
Gera curvas ROC/AUC por classe e tabela de métricas a partir de
um arquivo `predictions.csv` criado por `export_predictions_and_onnx.py`.

Saídas (salvas em `--results-dir`):
- roc_{class}.png (por classe)
- roc_curves.png (todas as curvas)
- metrics_table.csv
- metrics_table.md
- roc_aps.json (AUC e AP por classe)
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import label_binarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='results/presentation_best/predictions.csv')
    parser.add_argument('--results-dir', type=str, default='results/presentation_best')
    args = parser.parse_args()

    preds_path = args.predictions
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(preds_path)

    # detect prob_ columns and class names
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if len(prob_cols) == 0:
        raise SystemExit(f"Nenhuma coluna 'prob_' encontrada em {preds_path}")
    class_names = [c[len('prob_'):] for c in prob_cols]
    num_classes = len(class_names)

    y_true = df['true_label'].astype(int).values
    y_score = df[prob_cols].values
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    roc_aucs = {}
    ap_scores = {}

    # per-class ROC and AUC
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
        except ValueError:
            # e.g., if only one class present in y_true_bin[:, i]
            fpr, tpr, roc_auc = [0, 1], [0, 1], float('nan')

        roc_aucs[cname] = float(np.nan_to_num(roc_auc, nan=0.0))

        # per-class ROC figure
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_aucs[cname]:.3f}')
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {cname} (AUC={roc_aucs[cname]:.3f})')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'roc_{cname}.png'))
        plt.close()

        # average precision (AP)
        try:
            ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        except ValueError:
            ap = float('nan')
        ap_scores[cname] = float(np.nan_to_num(ap, nan=0.0))

    # aggregated ROC plot
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC={roc_aucs[cname]:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()

    # metrics table (precision, recall, f1, support)
    y_pred = df['pred_label'].astype(int).values
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    rows = []
    for i, cname in enumerate(class_names):
        rows.append({
            'class': cname,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'auc': float(roc_aucs.get(cname, 0.0)),
            'ap': float(ap_scores.get(cname, 0.0))
        })

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(results_dir, 'metrics_table.csv'), index=False)

    # also save a simple markdown table for inclusion in slides/TCC
    md_lines = [f"| Classe | Precision | Recall | F1 | Support | AUC | AP |",
                f"|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df_metrics.iterrows():
        md_lines.append(f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} | {r['auc']:.3f} | {r['ap']:.3f} |")
    with open(os.path.join(results_dir, 'metrics_table.md'), 'w', encoding='utf8') as f:
        f.write('\n'.join(md_lines))

    # save auc/ap dict
    with open(os.path.join(results_dir, 'roc_aps.json'), 'w', encoding='utf8') as f:
        json.dump({'auc': roc_aucs, 'ap': ap_scores}, f, indent=2)

    print('Saved ROC curves and metrics to', results_dir)


if __name__ == '__main__':
    main()
