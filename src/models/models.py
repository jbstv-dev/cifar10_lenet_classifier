import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, Model
from keras import optimizers


def lenet5(input_shape: tuple, output_shape: int) -> Model:
    """
    LeNet-5 model architecture.
    Arguments:
        input_shape: int -- dimensions of input image
        output_shape: int -- number of output classes

    Returns:
        model -- TF Keras model (object containing the information for the entire training process)
    """

    print("[Info] Building LeNet-5 model ...")
    model = Sequential()
    model.add(
        Conv2D(
            6,
            (5, 5),
            padding="valid",
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(
        Conv2D(
            16,
            (5, 5),
            padding="valid",
            activation="relu",
            kernel_initializer="he_normal",
        )
    )
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(84, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(output_shape, activation="softmax", kernel_initializer="he_normal"))
    sgd = optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    model.summary()
    return model


def AlexNet(input_shape: tuple, output_shape: int) -> Model:
    """
    AlexNet model architecture

    Arguments:
        input_shape: int -- dimensions of input image
        output_shape: int -- number of output classes

    Returns:
        model -- TF Keras model (object containing the information for the entire training process)
    """
    print("[Info] Building AlexNet model ...")
    model = Sequential()
    model.add(
        Conv2D(
            filters=96,
            kernel_size=(3, 3),
            strides=(4, 4),
            input_shape=input_shape,
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation="softmax"))

    print(model.summary())
    return model
