# README — train_melanoma.py

Este documento descreve o propósito, uso e comportamento do script `train_melanoma.py`, que implementa um pipeline de treinamento em PyTorch para classificação de imagens de lesões cutâneas (melanoma e outras classes).

Visão geral
---------

- Framework: PyTorch
- Modelos suportados: `resnet50` (padrão) e `efficientnet` (EfficientNet-B3 via `efficientnet_pytorch`).
- Tamanho de entrada: 224×224 (definido nos transforms)
- Estrutura de dados esperada: pastas organizadas por classe — ex.: `data/split/treino/<classe>/*.jpg` e `data/split/validacao/<classe>/*.jpg`.

Principais funcionalidades
-------------------------

- Transformações de treino/validação (resize, flip, rotation, normalize).
- Suporte a amostrador balanceado (`WeightedRandomSampler`) com smoothing (`inv`, `sqrt`, `none`).
- Losses: `CrossEntropyLoss` (com label smoothing opcional) e `FocalLoss` customizada (gamma, alpha).
- Mixup como técnica de regularização (`--mixup`).
- Treinamento em duas fases: cabeça (head) treinada inicialmente; depois fine-tuning do backbone em `--fine-tune-epoch`.
- Mixed precision (AMP) quando executado em GPU com `--mixed-precision`.
- Checkpointing (`last_checkpoint.pth`, `best_checkpoint.pth`, `checkpoint_epoch_*.pth`) e opção para retomar (`--resume`).
- Salvamento de métricas, relatórios e figuras em `results/`.

Dependências
------------

Instale as dependências num ambiente virtual. Atenção: `torch`/`torchvision` devem ser instalados conforme sua versão de CUDA (veja https://pytorch.org).

Exemplo de `requirements.txt` (já incluído no repositório):

```
torch
torchvision
efficientnet_pytorch
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
tqdm
pillow
```

Uso básico
---------

```powershell
python train_melanoma.py
```

Uso avançado (exemplo):
```powershell
python train_melanoma.py \
  --model resnet50 \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --finetune-lr 1e-5 \
  --fine-tune-epoch 5 \
  --train-dir data/split/treino \
  --val-dir data/split/validacao \
  --results-dir results \
  --models-dir models \
  --use-sampler \
  --mixed-precision \
  --save-epoch-checkpoints \
  --save-eval-per-epoch
```

Argumentos importantes (resumo)
--------------------------------

- `--model`: `resnet50` (padrão) ou `efficientnet`.
- `--batch-size`: tamanho do batch (padrão: 32).
- `--epochs`: número de épocas (padrão: 30).
- `--lr`: learning rate inicial (padrão: 1e-3).
- `--finetune-lr`: learning rate para fine-tuning (padrão: 1e-5).
- `--fine-tune-epoch`: época onde o script descongela o backbone e aplica `finetune-lr` (padrão: 5).
- `--patience`: paciência para early stopping (padrão: 8).
- `--train-dir` / `--val-dir`: diretórios de treino/val (padrões: `data/split/treino`, `data/split/validacao`).
- `--use-sampler`: habilita amostrador balanceado.
- `--sampler-smoothing`: `inv`, `sqrt` ou `none` (padrão: `inv`).
- `--loss`: `cross_entropy` (padrão) ou `focal` (use `--gamma` e `--focal-alpha` conforme necessário).
- `--label-smoothing`: valor de label smoothing para CrossEntropy (padrão: 0.0).
- `--mixup` / `--mixup-alpha`: ativa mixup (padrão alpha: 0.2).
- `--mixed-precision`: ativa AMP quando houver GPU.
- `--resume`: carrega `models/last_checkpoint.pth` para retomar treino.
- `--save-epoch-checkpoints`: salva checkpoints por época.
- `--save-eval-per-epoch`: salva relatório e matriz de confusão quando melhora performance.
- `--seed`: semente para reprodutibilidade (padrão: 42).

Comportamento do treino
----------------------

- O script calcula pesos de classe (`class_weights`) a partir da distribuição do conjunto de treino. Se `--use-sampler` estiver ativo, o sampler objetiva balancear amostras por classe; caso contrário, `class_weights` são passados para a `CrossEntropyLoss` quando aplicável.
- Inicialmente o backbone do modelo é congelado e apenas a cabeça (camadas finais) é treinada. Ao atingir `--fine-tune-epoch` todo o modelo é descongelado e o otimizador é substituído por um com `finetune-lr`.
- Early stopping é controlado pelo contador `--patience` em épocas sem melhoria na validação.

Detalhes de componentes
-----------------------

- `FocalLoss`: implementação custom que aceita `gamma` e `alpha`. `alpha` pode ser uma constante escalar ou um vetor de pesos por classe; se `alpha` for `None`, o fator é 1.0.
- Mixup: aplicado no batch com lambda ~ Beta(`mixup_alpha`, `mixup_alpha`) quando `--mixup` habilitado.
- AMP: quando `--mixed-precision` e GPU, usa `torch.cuda.amp.GradScaler()` e `autocast()` para acelerar.

Arquivos gerados
----------------

- Checkpoints: `models/last_checkpoint.pth`, `models/best_checkpoint.pth`, `models/checkpoint_epoch_*.pth` (se solicitado).
- Resultados e métricas: `results/history.json`, `results/report.txt`, `results/report.json`.
- Figuras: `results/accuracy.png`, `results/loss.png`, `results/confusion_matrix.png`, `results/confusion_matrix_normalized.png`.
- Relatórios por época (se `--save-eval-per-epoch`): `results/report_epoch_*.txt`, `results/report_epoch_*.json`, `results/confusion_epoch_*.png`.

Dicas práticas
-------------

- Para GPU, instale `torch`/`torchvision` apropriados à sua versão CUDA.
- Em CPU, evite `--mixed-precision` e reduza `--num-workers` se tiver problemas de I/O/memória.
- Se o dataset for muito desbalanceado, experimente `--use-sampler`, `--loss focal` e/ou `--alpha-balanced`.
- Para debug visual do fluxo de dados, verifique as amostras e classes carregadas por `torchvision.datasets.ImageFolder`.

Integração com pré-processamento
--------------------------------

É recomendado usar `predata.py` para pré-processar imagens (Dull Razor, CLAHE, redimensionamento) e gerar o split por `lesion_id` em `data/split/` antes do treino.

Exemplo de fluxo completo (PowerShell):

```powershell
# 1) Preprocessar e dividir
python predata.py --pasta-imagens archive/images --pasta-saida data/processado --csv archive/GroundTruth.csv --pasta-split data/split

# 2) Treinar
python train_melanoma.py --train-dir data/split/treino --val-dir data/split/validacao --model resnet50 --epochs 30 --batch-size 32 --use-sampler --mixed-precision
```

Problemas comuns
----------------

- Erro ao carregar `efficientnet_pytorch`: instale `efficientnet_pytorch` (`pip install efficientnet_pytorch`).
- `torch.cuda` não encontrado: verifique instalação do `torch` compatível com sua GPU/CUDA.
- Problemas com `label_smoothing` se sua versão do PyTorch é antiga — o script faz fallback caso a assinatura do construtor não aceite `label_smoothing`.

Licença
-------

Não foi encontrada uma licença explícita no repositório. Consulte o autor antes de redistribuir.

---

Se desejar, posso:

- Ajustar este README para gerar também um exemplo de `train` usando `efficientnet` (B3) e mostrar comandos de instalação `torch` com CUDA.
- Comitar os arquivos criados.
