"""
prepare_data_isic_gpu_ptbr.py
Pipeline completo ISIC com variáveis em português
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
# ====================== CONFIGURAÇÃO GPU ======================
def verificar_gpu():
    print(f"[INFO] OpenCV Version: {cv2.__version__}")
    try:
        quantidade = cv2.cuda.getCudaEnabledDeviceCount()
        if quantidade > 0:
            print(f"[OK] GPU detectada! Dispositivos CUDA: {quantidade}")
            print(f"[INFO] Dispositivo atual: {cv2.cuda.getDevice()}")
            return True
    except:
        pass
    
    print("[WARN] GPU nao disponivel. Executando em CPU.")
    return False

USAR_GPU = verificar_gpu()
# ====================== FUNÇÕES DE PROCESSAMENTO ======================
def remover_marcadores_coloridos(imagem_bgr):
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    mascaras = [
        cv2.inRange(hsv, np.array([85,40,40]), np.array([135,255,255])),
        cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255])),
        cv2.inRange(hsv, np.array([135,40,40]), np.array([165,255,255]))
    ]

    mascara_total = np.zeros_like(mascaras[0])
    for m in mascaras:
        mascara_total = cv2.bitwise_or(mascara_total, m)

    kernel = np.ones((7,7), np.uint8)
    mascara_total = cv2.morphologyEx(mascara_total, cv2.MORPH_OPEN, kernel)
    mascara_total = cv2.morphologyEx(mascara_total, cv2.MORPH_CLOSE, kernel)

    return cv2.inpaint(imagem_bgr, mascara_total, 10, cv2.INPAINT_TELEA)
def remover_pelos_dullrazor(imagem_bgr):
    imagem = remover_marcadores_coloridos(imagem_bgr)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, kernel)

    _, limiar = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    limiar = cv2.dilate(limiar, kernel, iterations=1)

    return cv2.inpaint(imagem, limiar, 9, cv2.INPAINT_TELEA)
def aplicar_clahe(imagem_bgr):
    lab = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_Lab2BGR)
def redimensionar_imagem(imagem, tamanho):
    if not USAR_GPU:
        return cv2.resize(imagem, tamanho)

    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(imagem)
        gpu = cv2.cuda.resize(gpu, tamanho)
        return gpu.download()
    except:
        return cv2.resize(imagem, tamanho)
# ====================== DEBUG ======================

def salvar_imagens_debug(imagem, caminho):
    nome = Path(caminho).stem
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite(f"debug/{nome}_original.jpg", imagem)
    dullrazor = remover_pelos_dullrazor(imagem)
    cv2.imwrite(f"debug/{nome}_dullrazor.jpg", dullrazor)
    clahe = aplicar_clahe(dullrazor)
    cv2.imwrite(f"debug/{nome}_clahe.jpg", clahe)

# ====================== PROCESSAMENTO ======================

def processar_imagem(caminho_origem, pasta_origem, pasta_destino,
                     usar_dullrazor=True,
                     usar_clahe=True,
                     tamanho=(224,224),
                     sobrescrever=False,
                     debug=False):

    try:
        caminho_relativo = os.path.relpath(caminho_origem, pasta_origem)
        caminho_destino = os.path.join(pasta_destino, caminho_relativo)

        os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)

        if os.path.exists(caminho_destino) and not sobrescrever:
            return True

        imagem = cv2.imread(caminho_origem)
        if imagem is None:
            return False

        if debug:
            salvar_imagens_debug(imagem, caminho_origem)

        if usar_dullrazor:
            imagem = remover_pelos_dullrazor(imagem)

        if usar_clahe:
            imagem = aplicar_clahe(imagem)

        imagem = redimensionar_imagem(imagem, tamanho)

        cv2.imwrite(caminho_destino, imagem)

        return True

    except Exception as erro:
        print(f"[ERRO] {caminho_origem}: {erro}")
        return False


def preprocessar_pasta(pasta_imagens, pasta_saida,
                       workers=4,
                       **kwargs):

    arquivos = [
        os.path.join(raiz, arquivo)
        for raiz, _, arquivos_lista in os.walk(pasta_imagens)
        for arquivo in arquivos_lista
        if arquivo.lower().endswith(('.jpg','.png','.jpeg'))
    ]

    print(f"[INFO] {len(arquivos)} imagens encontradas")

    if USAR_GPU:
        workers = min(workers, 2)

    processadas = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futuros = [
            executor.submit(processar_imagem, arq, pasta_imagens, pasta_saida, **kwargs)
            for arq in arquivos
        ]

        for futuro in as_completed(futuros):
            if futuro.result():
                processadas += 1

    print(f"[OK] {processadas}/{len(arquivos)} processadas")

    return pasta_saida

# ====================== SPLIT ======================

def construir_mapa_imagens(pasta, extensoes=('.jpg', '.jpeg', '.png')):
    """Constrói um dicionário map: stem -> caminho completo do arquivo de imagem.

    - `pasta`: caminho raiz onde procurar recursivamente
    - `extensoes`: tupla de extensões aceitas
    Retorna: dict {stem: caminho}
    """
    mapa = {}
    for raiz, _, arquivos in os.walk(pasta):
        for arq in arquivos:
            nome, ext = os.path.splitext(arq)
            if ext.lower() in extensoes and nome not in mapa:
                mapa[nome] = os.path.join(raiz, arq)
    return mapa


def dividir_por_lesion_id(caminho_csv, pasta_imagens, pasta_destino,
                           proporcao_validacao=0.15,
                           proporcao_teste=0.15):

    df = pd.read_csv(caminho_csv)

    gss = GroupShuffleSplit(n_splits=1, test_size=proporcao_teste)
    idx_treino, idx_teste = next(gss.split(df, groups=df['lesion_id']))

    df_treino = df.iloc[idx_treino]
    df_teste = df.iloc[idx_teste]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=proporcao_validacao)
    idx_treino, idx_val = next(gss_val.split(df_treino, groups=df_treino['lesion_id']))

    df_validacao = df_treino.iloc[idx_val]
    df_treino = df_treino.iloc[idx_treino]

    # mapeia arquivos de imagem (stem -> caminho) para busca eficiente (função no nível do módulo)
    conjuntos = {
        "treino": df_treino,
        "validacao": df_validacao,
        "teste": df_teste
    }

    imagens_map = construir_mapa_imagens(pasta_imagens)

    faltantes = 0
    copiados = 0

    for nome_conjunto, dados in conjuntos.items():
        for _, linha in dados.iterrows():
            key = str(linha['image'])
            origem = imagens_map.get(key)

            # preserva a extensão original quando disponível
            ext = ".jpg"
            if origem:
                try:
                    ext = Path(origem).suffix.lower() or ".jpg"
                except Exception:
                    ext = ".jpg"

            destino = os.path.join(pasta_destino, nome_conjunto, linha['dx'], f"{linha['image']}{ext}")
            os.makedirs(os.path.dirname(destino), exist_ok=True)

            if origem and os.path.exists(origem):
                img = cv2.imread(origem)
                if img is None:
                    print(f"[WARN] Nao foi possivel ler a imagem: {origem}")
                    faltantes += 1
                    continue
                cv2.imwrite(destino, img)
                copiados += 1
            else:
                faltantes += 1

    print(f"[OK] Divisao do dataset concluida — copiados: {copiados}, faltantes: {faltantes}, total esperado: {len(df)}")

# ====================== MAIN ======================

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--pasta-imagens", default="images")
    parser.add_argument("--pasta-saida", default="data/processado")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--pasta-split", default=None, help="Pasta destino para divisão (default: pasta-saida)")

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tamanho", type=int, default=224)

    parser.add_argument("--sem-dullrazor", action="store_true")
    parser.add_argument("--sem-clahe", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sobrescrever", action="store_true")

    args = parser.parse_args()

    pasta_processada = preprocessar_pasta(
        args.pasta_imagens,
        args.pasta_saida,
        workers=args.workers,
        usar_dullrazor=not args.sem_dullrazor,
        usar_clahe=not args.sem_clahe,
        tamanho=(args.tamanho, args.tamanho),
        sobrescrever=args.sobrescrever,
        debug=args.debug
    )

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"[ERROR] CSV nao encontrado: {args.csv}")
            return
        pasta_split = args.pasta_split or args.pasta_saida
        dividir_por_lesion_id(
            args.csv,
            pasta_processada,
            pasta_split
        )

if __name__ == "__main__":
    main()