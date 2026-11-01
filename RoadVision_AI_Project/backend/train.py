
"""
Training script.

Expected dataset:
backend/dataset/images/*.jpg|png
backend/dataset/masks/*.png

Each mask must be a binary image:
road = white (255)
background = black (0)

For best small-road performance, use high-resolution images/masks and
include many examples containing narrow residential/service roads.
"""
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from model import RoadUNet, IMG_SIZE

IMAGE_DIR = Path("dataset/images")
MASK_DIR = Path("dataset/masks")
MODEL_OUT = Path("models/road_unet.keras")

def load_pairs():
    pairs = []
    for img_path in sorted(IMAGE_DIR.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            continue
        mask_path = MASK_DIR / (img_path.stem + ".png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs

def preprocess_pair(img_path, mask_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Small morphological closing only for annotation gaps; avoid opening,
    # because opening can erase narrow roads.
    mask = (mask > 127).astype(np.float32)[..., None]
    img = img.astype(np.float32) / 255.0
    return img, mask

pairs = load_pairs()
if not pairs:
    raise RuntimeError(
        "No training pairs found. Put matching files in dataset/images and dataset/masks."
    )

split = int(len(pairs) * 0.8)
train_pairs = pairs[:split]
val_pairs = pairs[split:] or pairs[-1:]

def gen(items):
    while True:
        for ip, mp in items:
            yield preprocess_pair(ip, mp)

train_ds = tf.data.Dataset.from_generator(
    lambda: gen(train_pairs),
    output_signature=(
        tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
        tf.TensorSpec((IMG_SIZE, IMG_SIZE, 1), tf.float32),
    ),
).shuffle(32).batch(4).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: gen(val_pairs),
    output_signature=(
        tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
        tf.TensorSpec((IMG_SIZE, IMG_SIZE, 1), tf.float32),
    ),
).batch(4).prefetch(tf.data.AUTOTUNE)

steps = max(1, len(train_pairs) // 4)
val_steps = max(1, len(val_pairs) // 4)

model = RoadUNet.build()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT, monitor="val_loss", save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    steps_per_epoch=steps,
    validation_steps=val_steps,
    callbacks=callbacks,
)

MODEL_OUT.parent.mkdir(exist_ok=True)
model.save(MODEL_OUT)
print("Saved:", MODEL_OUT)
