# Correção SSL para download do EfficientNet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import json
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ========================= CONFIG =========================

BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCH = 5
PATIENCE = 8
TRAIN_DIR = "data/split/treino"
VAL_DIR = "data/split/validacao"
RESULTS_DIR = "results"
MODELS_DIR = "models"

device = None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True):
    try:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
    except TypeError:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is None:
            alpha_factor = 1.0
        else:
            if isinstance(self.alpha, (float, int)):
                alpha_factor = float(self.alpha)
            else:
                a = self.alpha.to(inputs.device)
                alpha_factor = a[targets]

        loss = - alpha_factor * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

# ========================= TRANSFORMS =========================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# ========================= MODEL =========================

def build_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    elif model_name == 'efficientnet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b3')
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")
    return model

# ========================= TRAIN =========================

def train_one_epoch(model, loader, criterion, optimizer, scaler=None, mixup_enabled=False, mixup_alpha=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if mixup_enabled and mixup_alpha > 0.0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = inputs.size(0)
            index = torch.randperm(batch_size).to(device)
            inputs_mixed = lam * inputs + (1 - lam) * inputs[index]
            labels_a, labels_b = labels, labels[index]

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs_mixed)
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs_mixed)
                loss_a = criterion(outputs, labels_a)
                loss_b = criterion(outputs, labels_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_a).sum().item()
            total += labels.size(0)
        else:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        pbar.set_postfix(loss=total_loss / total, acc=correct / total)
    return total_loss / total, correct / total

# ========================= VALIDATE =========================

def validate(model, loader, criterion, scaler=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=total_loss / total, acc=correct / total)
    return total_loss / total, correct / total, all_preds, all_labels

# ========================= PLOTS =========================

def plot_history(history, results_dir):
    plt.figure()
    plt.plot(history.get('train_acc', []), label='Train')
    plt.plot(history.get('val_acc', []), label='Validation')
    plt.legend()
    plt.title("Acurácia")
    plt.savefig(f"{results_dir}/accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.get('train_loss', []), label='Train')
    plt.plot(history.get('val_loss', []), label='Validation')
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"{results_dir}/loss.png")
    plt.close()

def plot_confusion(y_true, y_pred, class_names, results_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.close()

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão (normalizada)")
    plt.savefig(f"{results_dir}/confusion_matrix_normalized.png")
    plt.close()

# ========================= MAIN =========================

def main():
    global device
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet'])
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--finetune-lr', type=float, default=1e-5)
    parser.add_argument('--fine-tune-epoch', type=int, default=FINE_TUNE_EPOCH)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--train-dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--val-dir', type=str, default=VAL_DIR)
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-sampler', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--loss', choices=['cross_entropy', 'focal'], default='cross_entropy')
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--focal-alpha', type=float, default=None)
    parser.add_argument('--alpha-balanced', action='store_true')
    parser.add_argument('--sampler-smoothing', choices=['inv', 'sqrt', 'none'], default='inv')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-epoch-checkpoints', action='store_true')
    parser.add_argument('--save-eval-per-epoch', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results_dir = args.results_dir
    models_dir = args.models_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Modelo: {args.model}\n")

    if args.seed is not None:
        set_seed(args.seed)

    train_transform, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    print("Classes:", train_dataset.classes)
    print("Distribuição treino:", Counter(train_dataset.targets))
    print("Distribuição validação:", Counter(val_dataset.targets))
    num_classes = len(train_dataset.classes)

    counts = np.bincount(train_dataset.targets, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = len(train_dataset) / (num_classes * counts)
    class_weights = torch.tensor(weights, dtype=torch.float)

    sampler = None
    if args.use_sampler:
        if args.sampler_smoothing == 'inv':
            sample_weights_per_class = 1.0 / counts
        elif args.sampler_smoothing == 'sqrt':
            sample_weights_per_class = 1.0 / np.sqrt(counts)
        else:
            sample_weights_per_class = np.ones_like(counts, dtype=float)

        samples_weight = np.array([sample_weights_per_class[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == 'cuda'))

    model = build_model(args.model, num_classes).to(device)

    # Fase 1: congelar backbone
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar a cabeça
    if args.model == 'resnet50':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif args.model == 'efficientnet':
        for param in model._fc.parameters():
            param.requires_grad = True

    if args.loss == 'focal':
        if args.focal_alpha is not None:
            alpha = float(args.focal_alpha)
        elif args.alpha_balanced:
            alpha = class_weights / (class_weights.sum() + 1e-12)
        elif not args.use_sampler:
            alpha = class_weights / (class_weights.sum() + 1e-12)
        else:
            alpha = None
        criterion = FocalLoss(gamma=args.gamma, alpha=alpha)
    else:
        try:
            if not args.use_sampler:
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        except TypeError:
            if not args.use_sampler:
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            else:
                criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = make_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == 'cuda') else None

    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    if args.resume:
        ckpt_path = os.path.join(models_dir, 'last_checkpoint.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state_dict', {}))
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f'Warning: could not load optimizer state: {e}')
            start_epoch = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            print(f"Resuming from epoch {start_epoch}, best_acc={best_acc}")

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                                    mixup_enabled=args.mixup, mixup_alpha=args.mixup_alpha)
            val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, scaler)
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            with open(os.path.join(results_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            scheduler.step(val_loss)

            if epoch == args.fine_tune_epoch:
                print("→ Fine-tuning ativado")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
                scheduler = make_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

            last_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(last_ckpt, os.path.join(models_dir, 'last_checkpoint.pth'))
            if args.save_epoch_checkpoints:
                torch.save(last_ckpt, os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc
                }, os.path.join(models_dir, 'best_checkpoint.pth'))
                print("→ Melhor modelo salvo!")
                if args.save_eval_per_epoch:
                    try:
                        report_dict_epoch = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0, output_dict=True)
                        report_str_epoch = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0)
                        with open(os.path.join(results_dir, f'report_epoch_{epoch+1}.txt'), 'w') as f:
                            f.write(report_str_epoch)
                        with open(os.path.join(results_dir, f'report_epoch_{epoch+1}.json'), 'w') as f:
                            json.dump(report_dict_epoch, f, indent=2)
                        cm = confusion_matrix(labels, preds)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
                        plt.xlabel('Predito')
                        plt.ylabel('Real')
                        plt.title(f'Matriz de Confusão - Epoch {epoch+1}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(results_dir, f'confusion_epoch_{epoch+1}.png'))
                        plt.close()
                    except Exception as e:
                        print(f"Erro ao salvar avaliação: {e}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping!")
                    break
    except KeyboardInterrupt:
        print("\nTreinamento interrompido. Salvando...")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_acc': best_acc},
                   os.path.join(models_dir, 'interrupted_checkpoint.pth'))

    # Avaliação final
    best_path = os.path.join(models_dir, 'best_checkpoint.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    _, _, preds, labels = validate(model, val_loader, criterion, scaler)
    report_dict = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0, output_dict=True)
    report_str = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0)
    print("\nRelatório final (melhor checkpoint):\n", report_str)

    with open(os.path.join(results_dir, 'report.txt'), 'w') as f:
        f.write(report_str)
    with open(os.path.join(results_dir, 'report.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)

    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_history(history, results_dir)
    plot_confusion(labels, preds, train_dataset.classes, results_dir)

if __name__ == "__main__":
    main()