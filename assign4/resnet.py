import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, Activation, Add, Multiply,Input, ZeroPadding2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomFlip
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import HORIZONTAL
# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
def basic_blk(input, k, filter_depth, s):
    out = Conv2D(filter_depth,
                kernel_size=k[0],
                kernel_initializer='he_normal',  
                kernel_regularizer=regularizers.l2(l2=0.00001),
                padding='same',
                strides=s)(input)
                
    out = BatchNormalization(axis=3, momentum=0.9)(out)
    out = Activation('elu')(out)
    out = Conv2D(filter_depth,
                kernel_size=k[1],
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2=0.00001),
                padding='same')(out)

    out = BatchNormalization(axis=3, momentum=0.9)(out)
    return out

# https://www.cs.utah.edu/~srikumar/publications_files/epsilonResNet.pdf
def gate_blk(input, epsilon, L):
    ff_x = input
    x_left = Activation('relu')(1 * input - epsilon)
    x_right = Activation('relu')(-1 * input - epsilon)
    x = Add()([x_left, x_right])
    x = -1 * L * Activation('relu')(-1 * L * x + 1) + 1
    x = Activation('relu')(x)
    x = Multiply()([x, ff_x])
    return x

    

def ident_blk(input, filter_depth, subsample_stride):
    ff_input = input
    out = basic_blk(input, (3,3), filter_depth, subsample_stride)
    #out = gate_blk(out, 2.5, 100)
    out = Add()([out, ff_input])
    out = Activation('elu')(out)
    return out


def conv_blk(input, filter_depth, stride):
    ff_input = input
    out = basic_blk(input, (1,3), filter_depth, stride)
    ff_out = Conv2D(filter_depth,
                    kernel_size=(1,1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2=0.00001),
                    strides=stride,
                    padding='same')(ff_input)
    ff_out = BatchNormalization(axis=3)(ff_out)
    #out = gate_blk(out, 2.5, 100)
    out = Add()([out, ff_out])
    out = Activation('elu')(out)
    return out


def res_blk(x, filter_depth, num_layers):
    x = Conv2D(filter_depth,
                kernel_size=(1,1),
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2=0.00001),
                strides=1,
                padding='same')(x)
    for i in range(num_layers):
        x = ident_blk(x, filter_depth, 2)
    return x


def ResNet_N(in_shape, N):
    filter_depth = 64
    input = Input(in_shape)

    # Preprocessing method: RANDOM CROP & FLIP
    x = ZeroPadding2D(padding=(4,4))(input)
    x = RandomCrop(32, 32)(x)
    x = RandomFlip(mode=HORIZONTAL)(x)

    #x = input
    # model
    x = Conv2D(filter_depth,
               kernel_size=3,
               kernel_initializer='he_normal',
               padding='same',
               strides=2)(x)

    x = BatchNormalization(axis=3, momentum=0.9)(x)
    x = Activation('elu')(x)

    layers = [2] * N
    for i in range(layers[0]):
        x = ident_blk(x, filter_depth, 1)
        x = Dropout(0.3)(x)

    for i in range(len(layers[1:])):
        x = res_blk(x, (i + 2)*filter_depth, layers[i+1])
        x = Dropout(0.5)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.15)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=x, name=('ResNet-' + str(4*N + 2)))

    return model
    

