"""
DIAGNÓSTICO DE MELANOMAS UTILIZANDO REDES NEURAIS CONVOLUCIONAIS
Trabalho de Conclusão de Curso - Ciência da Computação
Autor: Diego da Silva Veloso de Faria
"""

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
from datetime import datetime
import json

# Helper to convert numpy types to JSON-serializable Python types
def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
# ========================= CONFIGURAÇÕES =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
# Por padrão, use as pastas geradas por `predata.py`:
DATASET_PATH = "data/treino"
# ========================= FUNÇÕES AUXILIARES =========================
def plot_history(history, model_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    h = history.history

    acc = h.get('accuracy', h.get('acc'))
    val_acc = h.get('val_accuracy', h.get('val_acc'))
    loss = h.get('loss')
    val_loss = h.get('val_loss')

    if acc is not None and val_acc is not None:
        axes[0].plot(acc, label='Train')
        axes[0].plot(val_acc, label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].legend()

    if loss is not None and val_loss is not None:
        axes[1].plot(loss, label='Train')
        axes[1].plot(val_loss, label='Validation')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_history.png')
    plt.close(fig)

def build_model(base_model_name, num_classes=2):
    if base_model_name == 'VGG16':
        base = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    elif base_model_name == 'ResNet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    elif base_model_name == 'InceptionV3':
        base = InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    else:
        raise ValueError("Modelo não suportado")
    
    base.trainable = False
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        output = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        output = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=loss, metrics=['accuracy'])
    return model

# ========================= DATA AUGMENTATION =========================
def train_all(dataset_train=DATASET_PATH, dataset_val=None, models_list=None, epochs=EPOCHS):
    """Treina modelos usando pastas organizadas por classe.

    - `dataset_train`: pasta com subpastas por classe (ex: data/treino)
    - `dataset_val`: se fornecido, usa pasta separada para validação (ex: data/validacao). Caso contrário, usa `validation_split`.
    """

    if models_list is None:
        models_list = ['VGG16', 'ResNet50', 'InceptionV3']

    # detecta classes pela presença de subpastas
    if not os.path.exists(dataset_train):
        raise FileNotFoundError(f"Pasta de treino nao encontrada: {dataset_train}")

    class_names = sorted([d for d in os.listdir(dataset_train) if os.path.isdir(os.path.join(dataset_train, d))])
    if len(class_names) == 0:
        raise ValueError(f"Nenhuma subpasta de classe encontrada em {dataset_train}")

    num_classes = len(class_names)
    class_mode = 'binary' if num_classes == 2 else 'categorical'

    print(f"Carregando dados. classes={class_names} (num_classes={num_classes})")

    if dataset_val is None:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_train,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=class_mode,
            subset='training',
            shuffle=True
        )

        val_generator = train_datagen.flow_from_directory(
            dataset_train,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=class_mode,
            subset='validation',
            shuffle=False
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2]
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            dataset_train,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=class_mode,
            shuffle=True
        )

        val_generator = test_datagen.flow_from_directory(
            dataset_val,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=class_mode,
            shuffle=False
        )

    results = {}
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    for model_name in models_list:
        print(f"\n{'='*60}")
        print(f"Treinando {model_name}...")
        print(f"{'='*60}")

        model = build_model(model_name, num_classes=num_classes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ModelCheckpoint(f'models/best_{model_name}.keras', monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Avaliação
        val_pred = model.predict(val_generator, verbose=0)
        if num_classes == 2:
            val_pred_class = (val_pred > 0.5).astype(int).flatten()
        else:
            val_pred_class = np.argmax(val_pred, axis=1)

        val_true = val_generator.classes

        # Métricas
        report = classification_report(val_true, val_pred_class, target_names=class_names, output_dict=True)

        recall_melanoma = None
        f1_melanoma = None
        if 'melanoma' in report:
            recall_melanoma = report['melanoma'].get('recall')
            f1_melanoma = report['melanoma'].get('f1-score')

        results[model_name] = {
            'accuracy': report.get('accuracy'),
            'recall_melanoma': recall_melanoma,
            'f1_melanoma': f1_melanoma,
            'history': history.history
        }

        # Salvar gráficos
        plot_history(history, model_name)

        # Matriz de confusão (headless)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(val_true, val_pred_class)
        fig = plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusao - {model_name}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(f'results/{model_name}_confusion_matrix.png')
        plt.close(fig)

        # salvar histórico e relatório de classificação em JSON
        with open(f'results/{model_name}_history.json', 'w') as f:
            json.dump(_to_serializable(history.history), f, indent=2)

        with open(f'results/{model_name}_classification_report.json', 'w') as f:
            json.dump(_to_serializable(report), f, indent=2)

    # retornar resultados agregados para uso posterior
    return results

def summarize_and_save(results_dict):
    print("\n" + "="*80)
    print("RESULTADOS COMPARATIVOS")
    print("="*80)

    comparison = pd.DataFrame({
        'Modelo': list(results_dict.keys()),
        'Acuracia': [results_dict[m]['accuracy'] for m in results_dict],
        'Recall Melanoma': [results_dict[m]['recall_melanoma'] for m in results_dict],
        'F1-Score Melanoma': [results_dict[m]['f1_melanoma'] for m in results_dict]
    })

    print(comparison.round(4))
    comparison.to_csv('results/comparacao_modelos.csv', index=False)

    if not comparison.empty and 'F1-Score Melanoma' in comparison:
        try:
            best_model = comparison.loc[comparison['F1-Score Melanoma'].idxmax()]
            print(f"\nMelhor modelo: {best_model['Modelo']} com F1-Score = {best_model['F1-Score Melanoma']:.4f}")
        except Exception:
            pass

    print("\nCodigo finalizado! Resultados salvos na pasta 'results/'")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Treinar modelos em pastas organizadas por classe')
    parser.add_argument('--train', default=DATASET_PATH, help='Pasta de treino (padrao: data/treino)')
    parser.add_argument('--val', default=None, help='Pasta de validacao (opcional). Se nao fornecido, usa validation_split de 0.2')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--models', nargs='*', default=None, help='Lista de modelos: VGG16 ResNet50 InceptionV3')

    args = parser.parse_args()
    results = train_all(dataset_train=args.train, dataset_val=args.val, models_list=args.models, epochs=args.epochs)
    # se train_all retornar um dict de resultados, salvamos o sumario
    if isinstance(results, dict):
        summarize_and_save(results)