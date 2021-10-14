from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Embedding,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
    Add,
    Input,
    GlobalAveragePooling1D,
)
from tensorflow.python.keras.layers.core import SpatialDropout1D
import numpy as np

def basic_blk(input, k, f, s):
    out = BatchNormalization(axis=1, momentum=0.9)(input)
    out = Activation("elu")(out)
    out = Conv1D(
        filters=f,
        kernel_size=k[0],
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2=0.0001),
        padding="same",
        strides=s,
    )(out)

    out = BatchNormalization(axis=1, momentum=0.9)(out)
    out = Activation("elu")(out)
    out = Conv1D(
        filters=f,
        kernel_size=k[1],
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2=0.0001),
        padding="same",
    )(out)

    return out


def ident_blk(input, filter_depth):
    ff_input = input
    out = Dropout(0.5)(input)
    out = basic_blk(out, (2, 2), filter_depth, (1, 1))
    out = Add()([out, ff_input])
    out = SpatialDropout1D(0.5)(out)
    return out


def conv_blk(input, filter_depth, stride):
    ff_input = input
    out = basic_blk(input, (2, 2), filter_depth, stride)
    ff_input = BatchNormalization(axis=1, momentum=0.9)(ff_input)
    ff_input = Activation("elu")(ff_input)
    ff_input = Conv1D(
        filter_depth,
        kernel_size=1,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2=0.00001),
        strides=stride,
        padding="same",
    )(ff_input)
    out = Add()([out, ff_input])
    return out


def res_blk(x, filter_depth, num_layers, init_stride):
    if init_stride != 1:
        num_layers -= 1
        x = conv_blk(x, filter_depth, init_stride)
    for i in range(num_layers):
        x = ident_blk(x, filter_depth)
    return x


def ResNet_N(doc_size, max_len, layers, classes):
    filter_depth = 16
    input = Input(max_len)
    x = input

    # model
    x = Embedding(input_dim = doc_size,
                  output_dim = 128)(x)
    x = Conv1D(
        filter_depth,
        kernel_size=2,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2=0.00001),
        padding="same",
        strides=2,
    )(x)

    x = BatchNormalization(axis=1, momentum=0.9)(x)
    x = Activation("elu")(x)

    x = res_blk(x, filter_depth, layers[0], init_stride=1)
    for i in range(len(layers[1:])):
        x = res_blk(x, (2 ** (i + 1)) * filter_depth, layers[i + 1], init_stride=2)

    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation="elu", kernel_initializer="he_normal")(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation="softmax", kernel_initializer="he_normal")(x)
    model = Model(
        inputs=input, outputs=x, name=("ResNet-" + str(2 * np.sum(layers) + 2))
    )

    return model