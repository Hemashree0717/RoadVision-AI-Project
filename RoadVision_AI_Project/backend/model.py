
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = 512

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);     p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256);    p4 = layers.MaxPooling2D()(c4)

    b = conv_block(p4, 512)
    b = layers.Dropout(0.25)(b)

    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(b)
    u4 = layers.Concatenate()([u4, c4]); d4 = conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d4)
    u3 = layers.Concatenate()([u3, c3]); d3 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2]); d2 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1]); d1 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return Model(inputs, outputs, name="RoadUNet")

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    denom = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)

def weighted_bce(y_true, y_pred, road_weight=4.0):
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    bce = -(y_true * tf.math.log(y_pred + eps) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred + eps))
    weights = 1.0 + (road_weight - 1.0) * y_true
    return tf.reduce_mean(weights * bce)

def road_loss(y_true, y_pred):
    # Dice keeps thin/rare road pixels important; weighted BCE reduces
    # the tendency to classify everything as background.
    return dice_loss(y_true, y_pred) + 0.5 * weighted_bce(y_true, y_pred)

class RoadUNet:
    @staticmethod
    def build():
        net = build_unet()
        net.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=road_loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return net

    @staticmethod
    def load(path: Path):
        return tf.keras.models.load_model(
            path,
            custom_objects={
                "road_loss": road_loss,
                "dice_loss": dice_loss,
                "weighted_bce": weighted_bce,
            },
        )

def preprocess(rgb):
    # Letterbox instead of stretching so road geometry is preserved.
    h, w = rgb.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y0 = (IMG_SIZE - nh) // 2
    x0 = (IMG_SIZE - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, (x0, y0, nw, nh), (w, h)

def postprocess(pred, geometry):
    x0, y0, nw, nh = geometry
    w, h = geometry[-1], geometry[-1]  # replaced below
    return pred

def predict_with_tiling(model, rgb, tile=512, overlap=128):
    """
    Sliding-window inference. This is important for small roads:
    a huge satellite image is not simply resized to 512x512, which would
    make narrow roads disappear.
    """
    h, w = rgb.shape[:2]
    stride = max(1, tile - overlap)
    accum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] != max(h - tile, 0): ys.append(max(h - tile, 0))
    if not xs or xs[-1] != max(w - tile, 0): xs.append(max(w - tile, 0))

    for y in ys:
        for x in xs:
            crop = rgb[y:y+tile, x:x+tile]
            ch, cw = crop.shape[:2]
            if ch != tile or cw != tile:
                pad = np.zeros((tile, tile, 3), dtype=np.uint8)
                pad[:ch, :cw] = crop
                crop = pad

            inp = crop.astype(np.float32) / 255.0
            pred = model.predict(inp[None, ...], verbose=0)[0, ..., 0]
            pred = pred[:ch, :cw]
            accum[y:y+ch, x:x+cw] += pred
            count[y:y+ch, x:x+cw] += 1.0

    return np.clip(accum / np.maximum(count, 1e-6), 0, 1)

def predict_mask(self, rgb):
    # Instance method is attached to keep the backend API simple.
    prob = predict_with_tiling(self, rgb)
    return (prob >= 0.35).astype(np.uint8) * 255

RoadUNet.predict_mask = predict_mask
