"""Quick check dos geradores usados por models.py
Imprime: classes encontradas, número de amostras e shape de um batch.
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = os.path.join('data', 'split', 'treino')
VAL_DIR = os.path.join('data', 'split', 'validacao')

print('Train dir:', TRAIN_DIR)
print('Val dir:', VAL_DIR)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print('Classes:', train_gen.class_indices)
print('Train samples:', train_gen.samples)
print('Val samples:', val_gen.samples)

x_batch, y_batch = next(train_gen)
print('Batch x shape:', x_batch.shape, 'dtype:', x_batch.dtype)
print('Batch y shape:', y_batch.shape, 'dtype:', y_batch.dtype)
