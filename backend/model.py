import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import numpy as np

from skimage.morphology import skeletonize


# ============================================================
# ROAD U-NET
# ============================================================

class RoadUNet:

    def __init__(self, model):
        self.model = model

    # ========================================================
    # BUILD U-NET
    # ========================================================

    @staticmethod
    def build(input_shape=(512, 512, 3)):

        inputs = keras.Input(shape=input_shape)

        # ====================================================
        # ENCODER
        # ====================================================

        c1 = layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        )(inputs)

        c1 = layers.BatchNormalization()(c1)

        c1 = layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        )(c1)

        p1 = layers.MaxPooling2D()(c1)

        # ----------------------------------------------------

        c2 = layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        )(p1)

        c2 = layers.BatchNormalization()(c2)

        c2 = layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        )(c2)

        p2 = layers.MaxPooling2D()(c2)

        # ----------------------------------------------------

        c3 = layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        )(p2)

        c3 = layers.BatchNormalization()(c3)

        c3 = layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        )(c3)

        p3 = layers.MaxPooling2D()(c3)

        # ----------------------------------------------------

        c4 = layers.Conv2D(
            256,
            3,
            padding="same",
            activation="relu"
        )(p3)

        c4 = layers.BatchNormalization()(c4)

        c4 = layers.Conv2D(
            256,
            3,
            padding="same",
            activation="relu"
        )(c4)

        p4 = layers.MaxPooling2D()(c4)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = layers.Conv2D(
            512,
            3,
            padding="same",
            activation="relu"
        )(p4)

        b = layers.BatchNormalization()(b)

        b = layers.Conv2D(
            512,
            3,
            padding="same",
            activation="relu"
        )(b)

        # ====================================================
        # DECODER
        # ====================================================

        u4 = layers.UpSampling2D()(b)

        u4 = layers.Concatenate()([
            u4,
            c4
        ])

        c5 = layers.Conv2D(
            256,
            3,
            padding="same",
            activation="relu"
        )(u4)

        c5 = layers.Conv2D(
            256,
            3,
            padding="same",
            activation="relu"
        )(c5)

        # ----------------------------------------------------

        u3 = layers.UpSampling2D()(c5)

        u3 = layers.Concatenate()([
            u3,
            c3
        ])

        c6 = layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        )(u3)

        c6 = layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        )(c6)

        # ----------------------------------------------------

        u2 = layers.UpSampling2D()(c6)

        u2 = layers.Concatenate()([
            u2,
            c2
        ])

        c7 = layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        )(u2)

        c7 = layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        )(c7)

        # ----------------------------------------------------

        u1 = layers.UpSampling2D()(c7)

        u1 = layers.Concatenate()([
            u1,
            c1
        ])

        c8 = layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        )(u1)

        c8 = layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        )(c8)

        # ====================================================
        # ROAD PROBABILITY
        # ====================================================

        outputs = layers.Conv2D(
            1,
            1,
            padding="same",
            activation="sigmoid"
        )(c8)

        return keras.Model(
            inputs,
            outputs,
            name="RoadVision_UNet"
        )

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    @classmethod
    def load(cls, path):

        model = keras.models.load_model(
            path,
            compile=False
        )

        return cls(model)

    # ========================================================
    # FAST ROAD PREDICTION
    # ========================================================

    def predict(self, rgb):
        """
        Road extraction from an RGB satellite image.

        Output:
            0   = black background
            255 = white roads
        """

        # ====================================================
        # SETTINGS
        # ====================================================

        TILE_SIZE = 512

        # 0.55 gives a good balance between missing roads
        # and unwanted white regions.
        THRESHOLD = 0.55

        # Keep small connected road segments.
        MIN_COMPONENT_SIZE = 8

        # Limit very large uploads for faster processing.
        MAX_SIZE = 1024

        # ====================================================
        # PREPARE INPUT
        # ====================================================

        image = np.asarray(rgb)

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]

        image = image.astype(np.uint8)

        original_h, original_w = image.shape[:2]

        # ====================================================
        # LIMIT VERY LARGE IMAGES
        # ====================================================

        scale = min(
            1.0,
            MAX_SIZE / max(
                original_h,
                original_w
            )
        )

        if scale < 1.0:

            new_w = int(
                original_w * scale
            )

            new_h = int(
                original_h * scale
            )

            image = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        work_h, work_w = image.shape[:2]

        # ====================================================
        # CREATE TILE GRID
        # ====================================================

        padded_h = int(
            np.ceil(
                work_h / TILE_SIZE
            ) * TILE_SIZE
        )

        padded_w = int(
            np.ceil(
                work_w / TILE_SIZE
            ) * TILE_SIZE
        )

        padded_image = np.zeros(
            (
                padded_h,
                padded_w,
                3
            ),
            dtype=np.uint8
        )

        padded_image[
            :work_h,
            :work_w
        ] = image

        # ====================================================
        # CREATE TILES
        # ====================================================

        tiles = []
        positions = []

        for y in range(
            0,
            padded_h,
            TILE_SIZE
        ):

            for x in range(
                0,
                padded_w,
                TILE_SIZE
            ):

                tile = padded_image[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ]

                tiles.append(tile)
                positions.append((y, x))

        # ====================================================
        # PREPARE BATCH
        # ====================================================

        batch = np.asarray(
            tiles,
            dtype=np.float32
        ) / 255.0

        # ====================================================
        # FAST BATCH PREDICTION
        # ====================================================

        predictions = self.model.predict(
            batch,
            verbose=0,
            batch_size=min(
                8,
                len(batch)
            )
        )

        predictions = np.asarray(
            predictions
        )

        # ====================================================
        # CREATE FULL PROBABILITY MAP
        # ====================================================

        probability = np.zeros(
            (
                padded_h,
                padded_w
            ),
            dtype=np.float32
        )

        count_map = np.zeros(
            (
                padded_h,
                padded_w
            ),
            dtype=np.float32
        )

        # ====================================================
        # MERGE TILE PREDICTIONS
        # ====================================================

        for i, (y, x) in enumerate(
            positions
        ):

            pred = predictions[i]

            if pred.ndim == 3:
                pred = pred[:, :, 0]

            probability[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += pred

            count_map[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += 1.0

        probability /= np.maximum(
            count_map,
            1e-6
        )

        # ====================================================
        # REMOVE PADDING
        # ====================================================

        probability = probability[
            :work_h,
            :work_w
        ]

        # ====================================================
        # RESIZE TO ORIGINAL SIZE
        # ====================================================

        if scale < 1.0:

            probability = cv2.resize(
                probability,
                (
                    original_w,
                    original_h
                ),
                interpolation=cv2.INTER_LINEAR
            )

        # ====================================================
        # LIGHT SMOOTHING
        # ====================================================

        probability = cv2.GaussianBlur(
            probability,
            (3, 3),
            0
        )

        # ====================================================
        # ROAD THRESHOLD
        # ====================================================

        binary = (
            probability >= THRESHOLD
        ).astype(
            np.uint8
        ) * 255

        # ====================================================
        # CONNECT SMALL ROAD GAPS
        # ====================================================

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        )

        # ====================================================
        # REMOVE TINY FALSE OBJECTS
        # ====================================================

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        cleaned = np.zeros_like(
            binary
        )

        for label in range(
            1,
            num_labels
        ):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= MIN_COMPONENT_SIZE:

                cleaned[
                    labels == label
                ] = 255

        # ====================================================
        # CONNECT ROAD SEGMENTS
        # ====================================================

        road_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            road_kernel,
            iterations=1
        )

        # ====================================================
        # CREATE THIN CONNECTED ROAD LINES
        # ====================================================

        skeleton = skeletonize(
            cleaned > 0
        )

        skeleton = (
            skeleton.astype(np.uint8)
            * 255
        )

        # ====================================================
        # MAKE THIN LINES CLEARLY VISIBLE
        # ====================================================

        line_kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3)
        )

        final_mask = cv2.dilate(
            skeleton,
            line_kernel,
            iterations=1
        )

        # ====================================================
        # FINAL BLACK AND WHITE MASK
        # ====================================================

        final_mask = np.where(
            final_mask > 0,
            255,
            0
        ).astype(
            np.uint8
        )

        return final_mask
