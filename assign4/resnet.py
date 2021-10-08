import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, GlobalAveragePooling2D, Cropping2D
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
def basic_blk(input, filter_depth, s):
    out = Conv2D(filter_depth,
                kernel_size=(3,3),
                kernel_initializer='he_normal',  
                kernel_regularizer=regularizers.l2(l2=0.00001),
                padding='same',
                strides=s)(input)
                
    out = BatchNormalization(axis=3, momentum=0.9)(out)
    out = Activation('elu')(out)
    out = Conv2D(filter_depth,
                kernel_size=(3,3),
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2=0.00001),
                padding='same')(out)

    out = BatchNormalization(axis=3, momentum=0.9)(out)
    return out

def ident_blk(input, filter_depth):
    ff_input = input
    out = basic_blk(input, filter_depth, (1,1))
    out = Add()([out, ff_input])
    out = Activation('elu')(out)
    return out


def conv_blk(input, filter_depth, stride):
    ff_input = input
    out = basic_blk(input, filter_depth, stride)
    out = Conv2D(4 * filter_depth,
                    kernel_size=(1,1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2=0.00005),
                    strides=stride,
                    padding='same')(out)
    out = BatchNormalization(axis=3)(out)
    out = Add()([out, ff_input])
    out = Activation('elu')(out)
    return out


def res_blk(ID, x, filter_depth, num_layers):
    x = conv_blk(x, ID * filter_depth, (2,2))
    for i in range(num_layers - 1):
        x = ident_blk(x, ID*filter_depth)
    return x


def ResNet_N(in_shape, N):
    filter_depth = 64
    input = Input(in_shape)

    # Preprocessing method: RANDOM CROP
    #x = ZeroPadding2D(padding=(4,4))(input)
    #x = RandomCrop(32, 32)(x)
    x = input
    # model
    x = Conv2D(filter_depth,
               kernel_size=(3,3),
               kernel_initializer='he_normal',
               padding='same',
               strides=(1,1))(x)

    x = BatchNormalization(axis=3, momentum=0.9)(x)
    x = Activation('elu')(x)

    layers = [2] * N
    for i in range(len(layers)):
        x = ident_blk(x, filter_depth)
    for i in range(len(layers[1:])):
        x = res_blk(i + 1, x, (i + 1)*filter_depth, layers[i])
    
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.15)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=x, name=('ResNet-' + str(4*N + 2)))

    return model
    

