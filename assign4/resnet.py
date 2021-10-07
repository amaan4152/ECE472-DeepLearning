import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, GlobalAveragePooling2D

# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
def VGG_blk(input, filter_depth, s):
    out = Conv2D(filter_depth,
                kernel_size=(3,3),
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2=0.0001),
                padding='same',
                strides=s)(input)
    out = BatchNormalization(axis=3, momentum=0.9)(out)
    out = Activation('elu')(out)
    out = Conv2D(filter_depth,
                kernel_size=(3,3),
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2=0.0001),
                padding='same')(out)

    out = BatchNormalization(axis=3, momentum=0.9)(out)
    return out

def ident_blk(input, filter_depth):
    ff_input = input
    out = VGG_blk(input, filter_depth, (1,1))
    out = Add()([out, ff_input])
    out = Activation('elu')(out)
    return out


def conv_blk(input, filter_depth, stride):
    ff_input = input
    out = VGG_blk(input, filter_depth, stride)
    ff_out = Conv2D(filter_depth,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2=0.0001),
                    strides=stride)(ff_input)
    ff_out = BatchNormalization(axis=3)(ff_out)
    out = Add()([out, ff_out])
    out = Activation('elu')(out)
    return out


def res_blk(ID, x, filter_depth, blk_depth):
    for i in range(blk_depth):
        x = conv_blk(x, ID * filter_depth, (2,2))
    return x


def ResNet_N(in_shape, N):
    filter_depth = 16

    input = Input(in_shape)

    # Preprocessing method: RANDOM CROP
    x = ZeroPadding2D(padding=(4,4))(input)

    x = Conv2D(16,
               kernel_size=(3,3),
               kernel_initializer='he_normal',
               padding='same')(x)

    x = BatchNormalization(axis=3, momentum=0.9)(x)
    x = Activation('elu')(x)
    layers = [2 * N] * 3
    for i in range(len(layers)):
        x = ident_blk(x, filter_depth)
    for i in range(len(layers)):
        x = res_blk(i + 2, x, (i + 2)*filter_depth, layers[i + 1])
    
    x = GlobalAveragePooling2D(padding='same')(x)
    x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1000, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(l2=0.005))(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=x, name=('ResNet-' + str(6*N+2)))

    return model
    

