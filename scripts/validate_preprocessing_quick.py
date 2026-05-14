"""Verificação rápida do pré-processamento:
- compara `archive/labels.csv` com arquivos em `data/processado` (existência)
- testa leitura e tamanho de uma amostra aleatória (até 50 imagens)
- imprime contagens por classe nas pastas `data/split`
"""
import os
import random
from pathlib import Path
import cv2
import pandas as pd

LABELS_CSV = Path('archive') / 'labels.csv'
PROCESSED_DIR = Path('data') / 'processado'
SPLIT_DIR = Path('data') / 'split'

print('Labels CSV:', LABELS_CSV)
print('Processed dir:', PROCESSED_DIR)
print('Split dir:', SPLIT_DIR)

labels = pd.read_csv(LABELS_CSV)
labels_set = set(labels['image'].astype(str))

images_map = {}
for root, _, files in os.walk(PROCESSED_DIR):
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in ('.jpg', '.jpeg', '.png'):
            if name not in images_map:
                images_map[name] = os.path.join(root, f)

print('\nTotal labels (rows):', len(labels))
print('Total arquivos encontrados em data/processado:', len(images_map))

# diferenças
missing = sorted([k for k in labels_set if k not in images_map])
extra = sorted([k for k in images_map.keys() if k not in labels_set])
print('\nFaltantes (labels sem arquivo):', len(missing))
print('Extras (arquivos sem label):', len(extra))
if missing:
    print('\nExemplos faltantes (max 20):')
    for k in missing[:20]:
        print('-', k)
if extra:
    print('\nExemplos extras em data/processado (max 20):')
    for k in extra[:20]:
        print('-', k)

# amostra para checar leitura e tamanho
sample_keys = list(labels_set & set(images_map.keys()))
random.seed(42)
M = min(50, len(sample_keys))
print(f'\nTestando leitura/shape de {M} imagens amostradas...')
if M > 0:
    sampled = random.sample(sample_keys, M)
else:
    sampled = []

not_readable = []
shape_mismatch = []

for k in sampled:
    p = images_map[k]
    img = cv2.imread(p)
    if img is None:
        not_readable.append(p)
        continue
    h, w = img.shape[:2]
    if (h, w) != (224, 224):
        shape_mismatch.append((k, p, (h, w)))

print('Nao-legiveis na amostra:', len(not_readable))
print('Tamanhos diferentes de 224x224 na amostra:', len(shape_mismatch))
if not_readable:
    print('\nExemplos nao-legiveis:')
    for p in not_readable[:10]:
        print('-', p)
if shape_mismatch:
    print('\nExemplos de mismatch de tamanho:')
    for s in shape_mismatch[:10]:
        print('-', s)

# contagens em data/split
print('\nContagens por classe em data/split:')
for conjunto in ['treino', 'validacao', 'teste']:
    path = SPLIT_DIR / conjunto
    if not path.exists():
        print(f'{conjunto}: nao existe ({path})')
        continue
    class_counts = {}
    for class_dir in sorted([d.name for d in path.iterdir() if d.is_dir()]):
        class_folder = path / class_dir
        n = 0
        for _root, _dirs, _files in os.walk(class_folder):
            for f in _files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    n += 1
        class_counts[class_dir] = n
    total = sum(class_counts.values())
    print(f'{conjunto}: total={total}; por-classe={class_counts}')

print('\nVerificacao rapida concluida.')
