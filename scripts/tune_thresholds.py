#!/usr/bin/env python3
"""
Ajuste de limiares por classe para maximizar F1 em um arquivo
`predictions.csv` (saída de `export_predictions_and_onnx.py`).

Gera em `--results-dir`:
- `predictions_thresholded.csv` (previsões usando limiares)
- `thresholds.json` (limiares por classe)
- `report_thresholded.txt` / `.json` (relatório de classificação)
- `confusion_matrix_thresholded.png` (matriz de confusão após limiares)

Regra de decisão: para cada amostra, seleciona classes com `prob >= threshold[class]`;
se houver candidatos, escolhe a com maior probabilidade entre eles; caso contrário,
usa `argmax` (fallback).
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support


def find_best_thresholds(y_true, y_scores, class_names, max_unique=1000):
    n_classes = y_scores.shape[1]
    thresholds = {}
    for i, cname in enumerate(class_names):
        scores = y_scores[:, i]
        unique_scores = np.unique(scores)
        if len(unique_scores) <= max_unique:
            candidates = unique_scores
        else:
            candidates = np.linspace(0.0, 1.0, 1001)

        best_f1 = -1.0
        best_t = 0.5
        y_true_bin = (y_true == i).astype(int)
        for t in candidates:
            y_pred_bin = (scores >= t).astype(int)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            if f1 > best_f1 or (f1 == best_f1 and t < best_t):
                best_f1 = f1
                best_t = float(t)

        thresholds[cname] = {'threshold': best_t, 'f1_binary': float(best_f1)}

    return thresholds


def apply_thresholds(y_scores, thresholds_list):
    # thresholds_list: array of shape (n_classes,)
    preds = []
    confidences = []
    for row in y_scores:
        candidates = np.where(row >= thresholds_list)[0]
        if len(candidates) > 0:
            chosen_idx = candidates[np.argmax(row[candidates])]
        else:
            chosen_idx = int(np.argmax(row))
        preds.append(int(chosen_idx))
        confidences.append(float(row[chosen_idx]))
    return np.array(preds), np.array(confidences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='results/presentation_best/predictions.csv')
    parser.add_argument('--results-dir', type=str, default='results/presentation_best')
    args = parser.parse_args()

    preds_path = args.predictions
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(preds_path)
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if len(prob_cols) == 0:
        raise SystemExit(f"Nenhuma coluna 'prob_' encontrada em {preds_path}")
    class_names = [c[len('prob_'):] for c in prob_cols]

    y_true = df['true_label'].astype(int).values
    y_scores = df[prob_cols].values

    # compute best thresholds per class
    thresholds = find_best_thresholds(y_true, y_scores, class_names)

    # apply thresholds to create new multiclass predictions
    thresholds_list = np.array([thresholds[c]['threshold'] for c in class_names], dtype=float)
    preds_thresh, conf_thresh = apply_thresholds(y_scores, thresholds_list)

    # save thresholds
    with open(os.path.join(results_dir, 'thresholds.json'), 'w', encoding='utf8') as f:
        json.dump(thresholds, f, indent=2)

    # add columns and save predictions
    df_out = df.copy()
    df_out['pred_label_thresholded'] = preds_thresh
    df_out['pred_name_thresholded'] = [class_names[i] for i in preds_thresh]
    df_out['pred_confidence_thresholded'] = conf_thresh
    df_out.to_csv(os.path.join(results_dir, 'predictions_thresholded.csv'), index=False)

    # classification report
    report_dict = classification_report(y_true, preds_thresh, target_names=class_names, zero_division=0, output_dict=True)
    report_str = classification_report(y_true, preds_thresh, target_names=class_names, zero_division=0)
    with open(os.path.join(results_dir, 'report_thresholded.json'), 'w', encoding='utf8') as f:
        json.dump(report_dict, f, indent=2)
    with open(os.path.join(results_dir, 'report_thresholded.txt'), 'w', encoding='utf8') as f:
        f.write(report_str)

    # confusion matrix image
    cm = confusion_matrix(y_true, preds_thresh)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (thresholded)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_thresholded.png'))
    plt.close()

    # compare macro F1 before/after
    orig_preds = df['pred_label'].astype(int).values
    try:
        f1_orig = f1_score(y_true, orig_preds, average='macro', zero_division=0)
        f1_new = f1_score(y_true, preds_thresh, average='macro', zero_division=0)
    except Exception:
        f1_orig = None
        f1_new = None

    summary = {
        'macro_f1_before': f1_orig,
        'macro_f1_after': f1_new,
        'thresholds_file': 'thresholds.json',
        'predictions_thresholded': 'predictions_thresholded.csv',
        'report_thresholded': 'report_thresholded.txt'
    }

    with open(os.path.join(results_dir, 'threshold_tuning_summary.json'), 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2)

    print('Threshold tuning completed. Summary saved to', os.path.join(results_dir, 'threshold_tuning_summary.json'))


if __name__ == '__main__':
    main()
