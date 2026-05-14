#!/usr/bin/env python3
"""
convert_groundtruth.py
Converte GroundTruth.csv (one-hot) para labels.csv com colunas: image,dx,lesion_id
Uso: python convert_groundtruth.py --input GroundTruth.csv --output labels.csv
"""
from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=None)
    parser.add_argument('--output', '-o', default=None)
    args = parser.parse_args()

    base = Path(__file__).parent
    input_path = Path(args.input) if args.input else base / 'GroundTruth.csv'
    output_path = Path(args.output) if args.output else base / 'labels.csv'

    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        return 2

    df = pd.read_csv(input_path)

    # rename common image id
    if 'image' not in df.columns and 'image_id' in df.columns:
        df = df.rename(columns={'image_id': 'image'})

    # if dx already present
    if 'dx' in df.columns:
        if 'lesion_id' not in df.columns:
            df['lesion_id'] = df['image']
        df[['image', 'dx', 'lesion_id']].to_csv(output_path, index=False)
        print(f"[OK] labels salvos em {output_path}")
        return 0

    # tentar detectar colunas one-hot conhecidas
    known = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    class_cols = [c for c in known if c in df.columns]

    # fallback: detectar colunas binárias 0/1 numéricas (exclui 'image')
    if not class_cols:
        possible = []
        for c in df.columns:
            if c in ('image', 'image_id'):
                continue
            try:
                s = df[c].dropna()
                if s.empty:
                    continue
                # checar se todos valores são 0/1 (int/float)
                if pd.api.types.is_numeric_dtype(s) and set(s.unique()).issubset({0, 1, 0.0, 1.0}):
                    possible.append(c)
            except Exception:
                continue
        class_cols = possible

    if not class_cols:
        print('[ERRO] Não foram encontradas colunas de classes nem coluna "dx" no CSV')
        return 3

    df['dx'] = df[class_cols].idxmax(axis=1)
    df['dx'] = df['dx'].astype(str).str.lower()

    if 'lesion_id' not in df.columns:
        df['lesion_id'] = df['image']

    df[['image', 'dx', 'lesion_id']].to_csv(output_path, index=False)
    print(f"[OK] labels salvos em {output_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
