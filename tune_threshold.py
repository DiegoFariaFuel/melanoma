import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report

# ------------------------------------------------------------
# Funções necessárias (evita dependência de train_melanoma.py)
# ------------------------------------------------------------
def get_transforms():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return None, val_transform   # só precisamos da val

def build_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )
    elif model_name == 'efficientnet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b3')
        in_features = model._fc.in_features
        model._fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")
    return model
# ------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_transform = get_transforms()
    val_dataset = datasets.ImageFolder("data/split/validacao", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ====================== CONFIGURAÇÃO ======================

    model_name = 'efficientnet'
    ckpt_path = "models_efficientnet_v3/best_checkpoint.pth"

    # ===========================================================

    model = build_model(model_name, 7).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_probs = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs)

    labels = np.array(val_dataset.targets)
    mel_idx = val_dataset.classes.index('mel')

    thresholds = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    for thr in thresholds:
        preds = np.argmax(all_probs, axis=1)
        mel_probs = all_probs[:, mel_idx]
        preds[mel_probs >= thr] = mel_idx
        print(f"\n{'='*40}")
        print(f"Threshold para melanoma: {thr}")
        print(classification_report(labels, preds, target_names=val_dataset.classes, digits=4))

if __name__ == "__main__":
    main()