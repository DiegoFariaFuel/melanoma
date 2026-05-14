#!/usr/bin/env python3
"""
scripts/list_data_full.py

Lista recursivamente o diretório de dados e imprime contagens por pasta e por classe.

Uso:
  python scripts/list_data_full.py --root data
"""

import os
import argparse

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


def is_image_file(fname):
    return os.path.splitext(fname.lower())[1] in IMG_EXTS


def count_files_in_dir(path):
    total = 0
    images = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.startswith('.'):
                continue
            total += 1
            if is_image_file(f):
                images += 1
    return total, images


def summarize(root):
    if not os.path.exists(root):
        print(f'Pasta não encontrada: {root}')
        return
    print(f'Analisando pasta: {root}\n')
    entries = sorted(os.listdir(root))
    total_all = 0
    total_images = 0

    for e in entries:
        path = os.path.join(root, e)
        if os.path.isdir(path):
            total, images = count_files_in_dir(path)
            total_all += total
            total_images += images
            print(f'{e}/: {total} arquivos ({images} imagens)')

            # listar nível abaixo (ex.: split -> treino/teste/validacao ou classes)
            subentries = sorted(os.listdir(path))
            subdirs = [s for s in subentries if os.path.isdir(os.path.join(path, s))]
            if subdirs:
                for sd in subdirs:
                    sd_path = os.path.join(path, sd)
                    sd_total, sd_images = count_files_in_dir(sd_path)
                    print(f'  {sd}/: {sd_total} arquivos ({sd_images} imagens)')

                    # se houver classes dentro deste subdiretório, listá-las
                    class_dirs = [c for c in sorted(os.listdir(sd_path)) if os.path.isdir(os.path.join(sd_path, c))]
                    if class_dirs:
                        for cls in class_dirs:
                            cls_path = os.path.join(sd_path, cls)
                            cls_total, cls_images = count_files_in_dir(cls_path)
                            print(f'    - {cls}: {cls_total} arquivos ({cls_images} imagens)')
        else:
            if e.startswith('.'):
                continue
            ext = os.path.splitext(e)[1].lower()
            print(f'{e} (arquivo) - ext={ext}')
            total_all += 1
            if is_image_file(e):
                total_images += 1

    print(f'\nTotal geral: {total_all} arquivos ({total_images} imagens)')


def main():
    parser = argparse.ArgumentParser(description='Lista o conteúdo da pasta de dados')
    parser.add_argument('--root', default='data', help='Pasta raiz (default: data)')
    args = parser.parse_args()
    summarize(args.root)


if __name__ == '__main__':
    main()
