"""Validações do pré-processamento:
- compara `archive/labels.csv` com arquivos em `data/processado`
- verifica imagens faltantes, não-legíveis e tamanhos diferentes de 224x224
- imprime contagens por classe nas pastas `data/split` (treino/validacao/teste)
"""
import os
from pathlib import Path
import cv2
import pandas as pd
from collections import Counter

LABELS_CSV = Path('archive') / 'labels.csv'
PROCESSED_DIR = Path('data') / 'processado'
SPLIT_DIR = Path('data') / 'split'

print('Labels CSV:', LABELS_CSV)
print('Processed dir:', PROCESSED_DIR)
print('Split dir:', SPLIT_DIR)

if not LABELS_CSV.exists():
    print('[ERROR] labels.csv nao encontrado em archive/. Saindo.')
    raise SystemExit(1)

if not PROCESSED_DIR.exists():
    print('[ERROR] pasta processada nao encontrada:', PROCESSED_DIR)
    raise SystemExit(2)

# carregar labels
labels = pd.read_csv(LABELS_CSV)
print('Total labels (rows):', len(labels))

# construir mapa de stem -> caminho
images_map = {}
for root, _, files in os.walk(PROCESSED_DIR):
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in ('.jpg', '.jpeg', '.png'):
            if name not in images_map:
                images_map[name] = os.path.join(root, f)

print('Total arquivos encontrados em data/processado:', len(images_map))

# verificar cobertura
missing = []
not_readable = []
shape_mismatch = []

for i, row in labels.iterrows():
    key = str(row['image'])
    path = images_map.get(key)
    if not path:
        missing.append(key)
        continue
    img = cv2.imread(path)
    if img is None:
        not_readable.append(path)
        continue
    h, w = img.shape[:2]
    if (h, w) != (224, 224):
        shape_mismatch.append((key, path, (h, w)))

print('\n=== Resumo da cobertura ===')
print('Labels total:', len(labels))
print('Encontradas:', len(labels) - len(missing))
print('Faltantes:', len(missing))
print('Nao-legiveis:', len(not_readable))
print('Tamanho diferente de 224x224:', len(shape_mismatch))

if missing:
    print('\nAmostra de filenames faltantes (max 20):')
    for k in missing[:20]:
        print('-', k)

if shape_mismatch:
    print('\nAmostra de mismatches de tamanho (max 10):')
    for s in shape_mismatch[:10]:
        print('-', s)

# checar contagens nas pastas de split
print('\n=== Contagens em data/split ===')
for conjunto in ['treino', 'validacao', 'teste']:
    path = SPLIT_DIR / conjunto
    if not path.exists():
        print(f'{conjunto}: nao existe ({path})')
        continue
    class_counts = {}
    # percorrer classes (subpastas)
    for class_name in sorted([d.name for d in path.iterdir() if d.is_dir()]):
        class_folder = path / class_name
        # contar arquivos recursivamente
        n = 0
        for _root, _dirs, _files in os.walk(class_folder):
            for f in _files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    n += 1
        class_counts[class_name] = n
    total = sum(class_counts.values())
    print(f'{conjunto}: total={total}; por-classe={class_counts}')

# imagens em processado que nao aparecem no CSV
extra = [k for k in images_map.keys() if k not in set(labels['image'].astype(str))]
print('\nArquivos em data/processado sem referencia no CSV (exemplos, max 20):', len(extra))
for e in extra[:20]:
    print('-', e)

print('\nValidação finalizada.')
