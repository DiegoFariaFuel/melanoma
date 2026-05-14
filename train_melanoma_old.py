import os
import time
import json
import random
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
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

# Initialize directories and device inside `main()` to avoid
# executing them in DataLoader worker subprocesses on Windows.
device = None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True):
    """Create ReduceLROnPlateau compatibly across torch versions.

    Some older torch versions do not accept `verbose` kwarg; try with it
    and fall back if TypeError is raised.
    """
    try:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
    except TypeError:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    Args:
        gamma (float): focusing parameter.
        alpha (float or Tensor or None): balancing factor. If Tensor, should have shape (C,).
        reduction: 'mean'|'sum'|'none'
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (B, C), targets: (B,)
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
                # ensure indexable by target
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

def build_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
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
            # approximate acc using labels_a
            correct += (preds == labels_a).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=total_loss / total, acc=correct / total)
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

def plot_history(history):
    # Accuracy
    plt.figure()
    plt.plot(history.get('train_acc', []), label='Train')
    plt.plot(history.get('val_acc', []), label='Validation')
    plt.legend()
    plt.title("Acurácia")
    plt.savefig(f"{RESULTS_DIR}/accuracy.png")
    plt.close()
    # Loss
    plt.figure()
    plt.plot(history.get('train_loss', []), label='Train')
    plt.plot(history.get('val_loss', []), label='Validation')
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"{RESULTS_DIR}/loss.png")
    plt.close()

def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.close()

    # normalized
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão (normalizada)")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_normalized.png")
    plt.close()

# ========================= MAIN =========================

def main():
    global device, RESULTS_DIR, MODELS_DIR
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--focal-alpha', type=float, default=None, help='Scalar alpha for focal loss (overrides balanced alpha)')
    parser.add_argument('--alpha-balanced', action='store_true', help='Use class-balanced alpha for focal loss')
    parser.add_argument('--sampler-smoothing', choices=['inv', 'sqrt', 'none'], default='inv', help='Weighting for sampler: inv=1/count, sqrt=1/sqrt(count), none=uniform')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation during training')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Alpha parameter for MixUp Beta distribution')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-epoch-checkpoints', action='store_true', help='Save checkpoint per epoch to models dir')
    parser.add_argument('--save-eval-per-epoch', action='store_true', help='Save evaluation (report + confusion) after each epoch')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    MODELS_DIR = args.models_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    if args.seed is not None:
        set_seed(args.seed)

    train_transform, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    print("Classes:", train_dataset.classes)
    print("Distribuição treino:", Counter(train_dataset.targets))
    print("Distribuição validação:", Counter(val_dataset.targets))
    num_classes = len(train_dataset.classes)

    # compute per-class counts and class weights (cpu tensors)
    counts = np.bincount(train_dataset.targets, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = len(train_dataset) / (num_classes * counts)
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Optionally oversample with WeightedRandomSampler
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

    model = build_model(num_classes).to(device)
    # 🔒 FASE 1: congelado
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Choose loss function
    if args.loss == 'focal':
        # determine alpha for focal loss
        if args.focal_alpha is not None:
            alpha = float(args.focal_alpha)
        elif args.alpha_balanced:
            alpha = class_weights.clone()
            alpha = alpha / (alpha.sum() + 1e-12)
        elif not args.use_sampler:
            alpha = class_weights.clone()
            alpha = alpha / (alpha.sum() + 1e-12)
        else:
            alpha = None
        criterion = FocalLoss(gamma=args.gamma, alpha=alpha)
    else:
        # CrossEntropy with optional label smoothing (if supported)
        try:
            if not args.use_sampler:
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        except TypeError:
            # older torch versions may not support label_smoothing
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

    # resume
    if args.resume:
        ckpt_path = os.path.join(MODELS_DIR, 'last_checkpoint.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state_dict', {}))
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print('Warning: could not load optimizer state (incompatible). Continuing without optimizer state.\n', e)
            start_epoch = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            print(f"Resuming from epoch {start_epoch}, best_acc={best_acc}")

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, mixup_enabled=args.mixup, mixup_alpha=args.mixup_alpha)
            val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, scaler)
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # save incremental history after each epoch
            try:
                with open(os.path.join(RESULTS_DIR, 'history.json'), 'w') as f:
                    json.dump(history, f, indent=2)
            except Exception:
                pass

            scheduler.step(val_loss)

            # 🔓 FASE 2: fine-tuning
            if epoch == args.fine_tune_epoch:
                print("→ Fine-tuning ativado")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
                scheduler = make_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

            # save last checkpoint
            last_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(last_ckpt, os.path.join(MODELS_DIR, 'last_checkpoint.pth'))
            # optionally keep per-epoch checkpoint (checkpoint_epoch_{n}.pth)
            if args.save_epoch_checkpoints:
                try:
                    torch.save(last_ckpt, os.path.join(MODELS_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
                except Exception:
                    pass

            # salvar melhor
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc
                }, os.path.join(MODELS_DIR, 'best_checkpoint.pth'))
                print("→ Melhor modelo salvo!")
                # also save evaluation artifacts for this epoch if requested
                if args.save_eval_per_epoch:
                    try:
                        report_dict_epoch = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0, output_dict=True)
                        report_str_epoch = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0)
                        with open(os.path.join(RESULTS_DIR, f'report_epoch_{epoch+1}.txt'), 'w') as f:
                            f.write(report_str_epoch)
                        with open(os.path.join(RESULTS_DIR, f'report_epoch_{epoch+1}.json'), 'w') as f:
                            json.dump(report_dict_epoch, f, indent=2)
                        # confusion matrix image
                        cm = confusion_matrix(labels, preds)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
                        plt.xlabel('Predito')
                        plt.ylabel('Real')
                        plt.title(f'Matriz de Confusão - Epoch {epoch+1}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_epoch_{epoch+1}.png'))
                        plt.close()
                    except Exception:
                        pass
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping!")
                    break
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário. Salvando checkpoint final...")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_acc': best_acc}, os.path.join(MODELS_DIR, 'interrupted_checkpoint.pth'))

    # ================= FINAL =================
    best_path = os.path.join(MODELS_DIR, 'best_checkpoint.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    _, _, preds, labels = validate(model, val_loader, criterion, scaler)
    report_dict = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0, output_dict=True)
    report_str = classification_report(labels, preds, target_names=train_dataset.classes, zero_division=0)
    print("\nRelatório:\n", report_str)
    with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w') as f:
        f.write(report_str)
    with open(os.path.join(RESULTS_DIR, 'report.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)

    # save history
    with open(os.path.join(RESULTS_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_history(history)
    plot_confusion(labels, preds, train_dataset.classes)

# ========================= RUN =========================

if __name__ == "__main__":
    main()