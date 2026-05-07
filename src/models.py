from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_1d_cnn(input_shape, num_classes: int):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn1d",
    )
    return model


def build_light_2d_cnn(input_shape, num_classes: int):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.SeparableConv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.SeparableConv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="light2dcnn")


def build_depthwise_2d_cnn(input_shape, num_classes: int):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, kernel_size=(1, 1), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="depthwise2dcnn")


def build_model(model_type: str, input_shape, num_classes: int):
    model_type = model_type.lower().strip()

    if model_type == "cnn1d":
        return build_1d_cnn(input_shape, num_classes)

    if model_type == "light2dcnn":
        return build_light_2d_cnn(input_shape, num_classes)

    if model_type == "depthwise2dcnn":
        return build_depthwise_2d_cnn(input_shape, num_classes)

    raise ValueError("Unknown model_type. Use one of: cnn1d, light2dcnn, depthwise2dcnn.")
