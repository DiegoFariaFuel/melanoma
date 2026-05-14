# README — predata.py

Este documento descreve o propósito, uso e detalhes de implementação do script `predata.py`, usado para pré-processamento e divisão (split) de conjuntos de imagens dermatoscópicas (ISIC / datasets similares).

**Visão geral**

O `predata.py` realiza duas tarefas principais:

- Pré-processamento de imagens (remoção de marcadores coloridos, remoção de pelos com Dull Razor, equalização de contraste via CLAHE, redimensionamento).
- Divisão do dataset por `lesion_id` (Group-aware split) para criar pastas `treino`, `validacao` e `teste` preservando lesões inteiras em um único conjunto.

O script tenta usar aceleração via OpenCV CUDA quando disponível, e executa processamento paralelo com `ThreadPoolExecutor` para acelerar o pipeline em CPUs multi-core.

Requisitos mínimos

- Python 3.8+
- numpy
- pandas
- scikit-learn
- opencv-python (ou uma build do OpenCV com suporte CUDA se você pretende usar GPU)

Sugestão de instalação (ambiente virtual):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy pandas scikit-learn opencv-python
```

Observação importante sobre GPU: as wheels oficiais do `opencv-python` normalmente NÃO incluem suporte CUDA. Para usar as rotinas `cv2.cuda.*` você precisa de uma build do OpenCV compilada com CUDA (instalada manualmente). O script detecta `cv2.cuda.getCudaEnabledDeviceCount()` e faz fallback automático para CPU se a GPU não estiver disponível.

Uso básico

```powershell
python predata.py --pasta-imagens archive/images --pasta-saida data/processado
```

Uso com CSV para divisão por `lesion_id` (ex.: `GroundTruth.csv` do ISIC):

```powershell
python predata.py --pasta-imagens archive/images \
  --pasta-saida data/processado \
  --csv archive/GroundTruth.csv \
  --pasta-split data/split \
  --workers 8 --tamanho 224
```

Argumentos disponíveis

- `--pasta-imagens` (default: `images`) — pasta raiz com imagens a serem processadas (procura recursiva).
- `--pasta-saida` (default: `data/processado`) — pasta onde as imagens processadas serão salvas, mantendo estrutura relativa.
- `--csv` (default: None) — arquivo CSV contendo metadados (é necessário para a etapa de split por lesão).
- `--pasta-split` (default: None) — local de saída para a divisão; por padrão usa `--pasta-saida` quando omitido.
- `--workers` (default: 4) — número de threads para processamento concorrente.
- `--tamanho` (default: 224) — tamanho final das imagens (quadrado).
- `--sem-dullrazor` — desabilita a etapa de remoção de pelos (Dull Razor).
- `--sem-clahe` — desabilita a etapa de CLAHE.
- `--debug` — salva imagens intermediárias (original, dullrazor, clahe) na pasta `debug/` para inspeção.
- `--sobrescrever` — sobrescrever arquivos já existentes na pasta de saída.

Como o pré-processamento funciona (funções principais)

- `verificar_gpu()` — checa `cv2.cuda.getCudaEnabledDeviceCount()` e define `USAR_GPU`. Se falhar ou não houver suporte, usa CPU.
- `remover_marcadores_coloridos(imagem_bgr)` — detecta e remove marcadores coloridos (range HSV) usando inpaint.
- `remover_pelos_dullrazor(imagem_bgr)` — aplica um pipeline tipo Dull Razor: blackhat + threshold + inpaint para reduzir pelos finos.
- `aplicar_clahe(imagem_bgr)` — converte para LAB e aplica CLAHE no canal L para melhorar contraste.
- `redimensionar_imagem(imagem, tamanho)` — usa `cv2.cuda` para redimensionar quando disponível; caso contrário `cv2.resize` normal.
- `salvar_imagens_debug(imagem, caminho)` — salva versões intermediárias em `debug/` para análise visual.
- `processar_imagem(caminho_origem, pasta_origem, pasta_destino, ...)` — pipeline que aplica as transformações e grava o arquivo destino preservando a estrutura relativa.
- `preprocessar_pasta(pasta_imagens, pasta_saida, workers, **kwargs)` — varre recursivamente as imagens e processa em paralelo com `ThreadPoolExecutor`.

Split por `lesion_id` — preservando lesões

Função: `dividir_por_lesion_id(caminho_csv, pasta_imagens, pasta_destino, proporcao_validacao=0.15, proporcao_teste=0.15)`

- Exige um CSV com, no mínimo, as colunas: `lesion_id`, `image`, `dx`.
  - `lesion_id`: identificador da lesão (usado para agrupar amostras da mesma lesão).
  - `image`: nome do arquivo (sem extensão) usado como chave para localizar o arquivo processado.
  - `dx`: rótulo/diagnóstico (ex.: `melanoma`, `nevus`, etc.) — usado para criar subpastas por classe.
- O algoritmo usa `GroupShuffleSplit` para primeiro separar `teste` (por padrão 15%), e depois divide o restante entre `treino` / `validacao` usando outra divisão por grupos (val 15% do resto). Isso evita que imagens da mesma lesão apareçam em conjuntos diferentes.
- As imagens são copiadas para `pasta_destino/<conjunto>/<dx>/<image>.<ext>` preservando a extensão original quando possível.
- O script contabiliza e reporta imagens faltantes.

Saídas geradas

- `--pasta-saida` (ex.: `data/processado`) — imagens processadas (mesma árvore relativa das imagens de entrada).
- `--pasta-split` (ex.: `data/split`) — subpastas `treino`, `validacao`, `teste` organizadas por classe (`dx`).
- `debug/` — imagens intermediárias (se `--debug` foi passado).

Boas práticas e dicas

- Garanta que os nomes na coluna `image` do CSV correspondam ao nome do arquivo sem extensão (o script mapeia por `stem`).
- Se houver muitas imagens e o seu OpenCV não tem CUDA, use `--workers` alto mas cuidado com I/O e memória.
- Para uso de GPU em redimensionamento, instale uma build do OpenCV compilada com CUDA (não é o padrão do pip).
- Use `--debug` em um subconjunto pequeno para inspecionar o efeito do Dull Razor e CLAHE antes de processar todo o dataset.
- Se as classes estiverem desbalanceadas, o split por `lesion_id` ainda preserva proporções globais mas pode haver variação; verifique a distribuição nas pastas resultantes.

Erros comuns e resolução rápida

- Imagens não encontradas na etapa de split: verifique se os stems no CSV batem com os nomes de arquivo (sem extensão) e se `preprocessar_pasta` já foi executado sobre as imagens.
- `cv2.cuda` não encontrado: provável que sua instalação do OpenCV não tenha suporte CUDA; o script faz fallback para CPU.
- `imagem is None` ao ler: arquivo corrompido ou formato não suportado; o arquivo é contabilizado como faltante.

Integração com o fluxo de treino

Fluxo recomendado:

1. Executar `preprocessar_pasta` para limpar, equalizar e redimensionar as imagens.
2. Executar `dividir_por_lesion_id` informando o CSV para gerar `data/split/treino` e `data/split/validacao`.
3. Rodar o treinamento com `train_melanoma.py` (ex.: `python train_melanoma.py --train-dir data/split/treino --val-dir data/split/validacao`).

Exemplo rápido (Windows PowerShell):

```powershell
# 1) preprocessar
python predata.py --pasta-imagens archive/images --pasta-saida data/processado --workers 8 --tamanho 224

# 2) dividir por lesion_id
python predata.py --pasta-imagens data/processado --csv archive/GroundTruth.csv --pasta-split data/split

# 3) treinar
python train_melanoma.py --train-dir data/split/treino --val-dir data/split/validacao
```

Possíveis melhorias

- Expor as proporções de `validacao` e `teste` via argumentos CLI.
- Adicionar um parâmetro `--seed` para tornar as divisões reproducíveis.
- Salvar um CSV com o mapeamento final (`conjunto,image,path,dx,lesion_id`) para auditoria.

Autor / Créditos

O script contém rotinas comumente usadas em pipelines de processamento dermatoscópico (Dull Razor, CLAHE). Revise e cite adequadamente se for parte de trabalho acadêmico ou publicação.

Licença

Não há licença explícita no repositório. Verifique com o autor antes de redistribuir.

---

Se quiser, eu posso também:

- Gerar um `requirements.txt` contendo as dependências exatas usadas.
- Adicionar um CSV de mapeamento final após o split.
- Comitar esse README no repositório.
