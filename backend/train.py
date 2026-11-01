import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# ROADVISION AI - IMPROVED ROAD U-NET TRAINING
# ============================================================
#
# IMPORTANT:
# This model learns ROAD vs NON-ROAD from the masks in the
# dataset. It does NOT have a separate "vehicle road" label.
#
# The changes below reduce false positives such as buildings,
# fields and other large areas by:
#   1. Using more no-road / hard-negative patches.
#   2. Reducing the road class weight.
#   3. Using Dice + weighted BCE loss.
#   4. Keeping road-focused patches for small roads.
#   5. Using precision and IoU during validation.
#
# ============================================================


# -----------------------------
# SETTINGS
# -----------------------------

IMG_SIZE = 512
BATCH_SIZE = 4

EPOCHS = 40

STEPS_PER_EPOCH = 1000
VALIDATION_STEPS = 200

# IMPORTANT:
# The old value was 0.80.
# That gave the model too many road-containing patches and
# not enough pure-background examples.
ROAD_PATCH_PROBABILITY = 0.65

# A patch is considered a road patch when at least this
# percentage of pixels belong to roads.
MIN_ROAD_RATIO = 0.005

# Random seed
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "dataset" / "images"
MASK_DIR = BASE_DIR / "dataset" / "masks"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "road_unet.keras"


print("=" * 70)
print("RoadVision AI - Improved Road U-Net Training")
print("=" * 70)

print(f"Image directory : {IMAGE_DIR}")
print(f"Mask directory  : {MASK_DIR}")
print(f"Model output    : {MODEL_PATH}")
print()


# ============================================================
# FIND DATASET
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob("*_sat.jpg")
)

if len(image_files) == 0:

    raise RuntimeError(
        "No satellite images found. "
        "Please run prepare_dataset.py first."
    )


pairs = []

for image_path in image_files:

    image_id = image_path.stem.replace(
        "_sat",
        ""
    )

    mask_path = MASK_DIR / f"{image_id}_mask.png"

    if mask_path.exists():

        pairs.append(
            (
                image_path,
                mask_path
            )
        )


print(
    f"Total image-mask pairs: {len(pairs)}"
)

if len(pairs) < 10:

    raise RuntimeError(
        "Not enough image-mask pairs found."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

random.shuffle(pairs)

validation_count = max(
    1,
    int(len(pairs) * 0.15)
)

validation_pairs = pairs[
    :validation_count
]

training_pairs = pairs[
    validation_count:
]

print(
    f"Training images   : {len(training_pairs)}"
)

print(
    f"Validation images : {len(validation_pairs)}"
)

print()


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_and_mask(
    image_path,
    mask_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: {image_path}"
        )


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise ValueError(
            f"Could not read mask: {mask_path}"
        )


    # DeepGlobe masks may not always be exactly 0/255.
    # Road = 1, background = 0.
    mask = (
        mask >= 128
    ).astype(
        np.float32
    )


    return image, mask


# ============================================================
# RANDOM PATCH EXTRACTION
# ============================================================

def get_random_patch(
    image,
    mask,
    prefer_road=True
):

    height, width = mask.shape


    # --------------------------------------------------------
    # SMALL IMAGE
    # --------------------------------------------------------

    if (
        height < IMG_SIZE
        or width < IMG_SIZE
    ):

        image = cv2.resize(
            image,
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        return image, mask


    last_image_patch = None
    last_mask_patch = None


    # --------------------------------------------------------
    # ROAD PATCH
    # --------------------------------------------------------

    if prefer_road:

        for _ in range(30):

            x = random.randint(
                0,
                width - IMG_SIZE
            )

            y = random.randint(
                0,
                height - IMG_SIZE
            )

            image_patch = image[
                y:y + IMG_SIZE,
                x:x + IMG_SIZE
            ]

            mask_patch = mask[
                y:y + IMG_SIZE,
                x:x + IMG_SIZE
            ]

            last_image_patch = image_patch
            last_mask_patch = mask_patch

            road_pixels = np.mean(
                mask_patch
            )

            if road_pixels > MIN_ROAD_RATIO:

                return (
                    image_patch,
                    mask_patch
                )


    # --------------------------------------------------------
    # HARD NEGATIVE / BACKGROUND PATCH
    # --------------------------------------------------------
    #
    # This is important for stopping houses, fields and other
    # background regions from becoming white.
    # --------------------------------------------------------

    else:

        for _ in range(30):

            x = random.randint(
                0,
                width - IMG_SIZE
            )

            y = random.randint(
                0,
                height - IMG_SIZE
            )

            image_patch = image[
                y:y + IMG_SIZE,
                x:x + IMG_SIZE
            ]

            mask_patch = mask[
                y:y + IMG_SIZE,
                x:x + IMG_SIZE
            ]

            last_image_patch = image_patch
            last_mask_patch = mask_patch

            road_pixels = np.mean(
                mask_patch
            )

            if road_pixels <= MIN_ROAD_RATIO:

                return (
                    image_patch,
                    mask_patch
                )


    return (
        last_image_patch,
        last_mask_patch
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment(
    image,
    mask
):

    # Horizontal flip
    if random.random() < 0.5:

        image = np.fliplr(
            image
        )

        mask = np.fliplr(
            mask
        )


    # Vertical flip
    if random.random() < 0.5:

        image = np.flipud(
            image
        )

        mask = np.flipud(
            mask
        )


    # 90-degree rotation
    if random.random() < 0.5:

        k = random.randint(
            1,
            3
        )

        image = np.rot90(
            image,
            k
        )

        mask = np.rot90(
            mask,
            k
        )


    # Small brightness change
    if random.random() < 0.35:

        factor = random.uniform(
            0.85,
            1.15
        )

        image = np.clip(
            image.astype(
                np.float32
            ) * factor,
            0,
            255
        ).astype(
            np.uint8
        )


    return (
        image.copy(),
        mask.copy()
    )


# ============================================================
# DATA GENERATOR
# ============================================================

class RoadPatchGenerator(
    keras.utils.Sequence
):

    def __init__(
        self,
        data_pairs,
        batch_size,
        steps_per_epoch,
        training=True
    ):

        self.data_pairs = data_pairs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.training = training


    def __len__(self):

        return self.steps_per_epoch


    def __getitem__(
        self,
        index
    ):

        images = []
        masks = []


        for _ in range(
            self.batch_size
        ):

            image_path, mask_path = random.choice(
                self.data_pairs
            )


            image, mask = load_image_and_mask(
                image_path,
                mask_path
            )


            # ------------------------------------------------
            # TRAINING PATCH MIX
            # ------------------------------------------------
            #
            # 65% road-containing patches
            # 35% hard-negative/background patches
            #
            # This is the key change for reducing false
            # detections on houses and fields.
            # ------------------------------------------------

            prefer_road = (
                self.training
                and random.random()
                < ROAD_PATCH_PROBABILITY
            )


            image, mask = get_random_patch(
                image,
                mask,
                prefer_road=prefer_road
            )


            if self.training:

                image, mask = augment(
                    image,
                    mask
                )


            # Normalize image
            image = (
                image.astype(
                    np.float32
                ) / 255.0
            )


            mask = mask.astype(
                np.float32
            )


            # Add channel to mask
            mask = np.expand_dims(
                mask,
                axis=-1
            )


            images.append(
                image
            )

            masks.append(
                mask
            )


        return (
            np.array(
                images,
                dtype=np.float32
            ),
            np.array(
                masks,
                dtype=np.float32
            )
        )


# ============================================================
# U-NET
# ============================================================

def conv_block(
    x,
    filters
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Activation(
        "relu"
    )(x)


    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Activation(
        "relu"
    )(x)


    return x


def build_unet():

    inputs = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )


    # -----------------------------
    # ENCODER
    # -----------------------------

    c1 = conv_block(
        inputs,
        32
    )

    p1 = layers.MaxPooling2D(
        (2, 2)
    )(c1)


    c2 = conv_block(
        p1,
        64
    )

    p2 = layers.MaxPooling2D(
        (2, 2)
    )(c2)


    c3 = conv_block(
        p2,
        128
    )

    p3 = layers.MaxPooling2D(
        (2, 2)
    )(c3)


    c4 = conv_block(
        p3,
        256
    )

    p4 = layers.MaxPooling2D(
        (2, 2)
    )(c4)


    # -----------------------------
    # BOTTLENECK
    # -----------------------------

    c5 = conv_block(
        p4,
        512
    )


    # -----------------------------
    # DECODER
    # -----------------------------

    u6 = layers.Conv2DTranspose(
        256,
        2,
        strides=2,
        padding="same"
    )(c5)

    u6 = layers.Concatenate()(
        [
            u6,
            c4
        ]
    )

    c6 = conv_block(
        u6,
        256
    )


    u7 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(c6)

    u7 = layers.Concatenate()(
        [
            u7,
            c3
        ]
    )

    c7 = conv_block(
        u7,
        128
    )


    u8 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(c7)

    u8 = layers.Concatenate()(
        [
            u8,
            c2
        ]
    )

    c8 = conv_block(
        u8,
        64
    )


    u9 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c8)

    u9 = layers.Concatenate()(
        [
            u9,
            c1
        ]
    )

    c9 = conv_block(
        u9,
        32
    )


    # -----------------------------
    # OUTPUT
    # -----------------------------

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c9)


    model = keras.Model(
        inputs,
        outputs,
        name="RoadVision_UNet"
    )


    return model


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    y_true,
    y_pred
):

    smooth = 1.0

    y_true_f = tf.reshape(
        y_true,
        [-1]
    )

    y_pred_f = tf.reshape(
        y_pred,
        [-1]
    )

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        tf.reduce_sum(
            y_true_f
        )
        + tf.reduce_sum(
            y_pred_f
        )
        + smooth
    )

    return 1.0 - dice


# ============================================================
# WEIGHTED BINARY CROSS ENTROPY
# ============================================================

def weighted_bce(
    y_true,
    y_pred
):

    # Old value was 4.0.
    # Lowering it reduces the tendency to turn large
    # non-road areas into white predictions.
    road_weight = 2.0


    bce = keras.backend.binary_crossentropy(
        y_true,
        y_pred
    )


    weights = (
        1.0
        + y_true * (
            road_weight - 1.0
        )
    )


    return tf.reduce_mean(
        bce * weights
    )


# ============================================================
# COMBINED LOSS
# ============================================================

def road_loss(
    y_true,
    y_pred
):

    return (
        dice_loss(
            y_true,
            y_pred
        )
        + 0.5 * weighted_bce(
            y_true,
            y_pred
        )
    )


# ============================================================
# METRICS
# ============================================================

def dice_metric(
    y_true,
    y_pred
):

    y_pred = tf.cast(
        y_pred > 0.50,
        tf.float32
    )


    y_true_f = tf.reshape(
        y_true,
        [-1]
    )

    y_pred_f = tf.reshape(
        y_pred,
        [-1]
    )


    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )


    return (
        2.0 * intersection
        + 1.0
    ) / (
        tf.reduce_sum(
            y_true_f
        )
        + tf.reduce_sum(
            y_pred_f
        )
        + 1.0
    )


def road_iou(
    y_true,
    y_pred
):

    y_pred = tf.cast(
        y_pred > 0.50,
        tf.float32
    )


    intersection = tf.reduce_sum(
        y_true * y_pred
    )


    union = (
        tf.reduce_sum(
            y_true
        )
        + tf.reduce_sum(
            y_pred
        )
        - intersection
    )


    return (
        intersection + 1.0
    ) / (
        union + 1.0
    )


# ============================================================
# PRECISION
# ============================================================

road_precision = keras.metrics.Precision(
    name="road_precision",
    thresholds=0.50
)


road_recall = keras.metrics.Recall(
    name="road_recall",
    thresholds=0.50
)


# ============================================================
# CREATE DATA GENERATORS
# ============================================================

print(
    "Creating training generator..."
)


train_generator = RoadPatchGenerator(
    training_pairs,
    batch_size=BATCH_SIZE,
    steps_per_epoch=STEPS_PER_EPOCH,
    training=True
)


validation_generator = RoadPatchGenerator(
    validation_pairs,
    batch_size=BATCH_SIZE,
    steps_per_epoch=VALIDATION_STEPS,
    training=False
)


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "Building U-Net..."
)


model = build_unet()


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss=road_loss,
    metrics=[
        dice_metric,
        road_iou,
        road_precision,
        road_recall
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_road_iou",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# START TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

print(
    "Road patches       : 65%"
)

print(
    "Hard-negative patches: 35%"
)

print(
    "Road weight        : 2.0"
)

print()
print(
    "The model is being trained to reduce false road"
)

print(
    "detections in buildings, fields and other background."
)

print()


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    str(MODEL_PATH)
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print()

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()

print(
    "Replace the old road_unet.keras with this model "
    "and restart the FastAPI backend."
)

print("=" * 70)
