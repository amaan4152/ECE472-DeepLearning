import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D
from tensorflow.python.keras.layers.convolutional import Conv

# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
# https://arxiv.org/pdf/1512.03385.pdf
def VGG_blk(input, filter_depth, s):
    out = Conv2D(filter_depth,
                kernel_size=(1,1),
                kernel_initializer='he_normal',
                padding='same',
                strides=s)(input)

    out = BatchNormalization(axis=3)(out)
    out = Activation('elu')(out)
    out = Conv2D(filter_depth,
                kernel_size=(3,3),
                kernel_initializer='he_normal',
                padding='same')(out)

    out = BatchNormalization(axis=3)(out)
    out = Activation('elu')(out)
    out = Conv2D(4 * filter_depth,
                kernel_size=(1,1),
                kernel_initializer='he_normal',
                padding='same',
                strides=s)(input)

    out = BatchNormalization(axis=3)(out)
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
    ff_out = Conv2D(4 * filter_depth,
                    kernel_size=(1,1),
                    kernel_initializer='he_normal',
                    strides=stride)(ff_input)
    ff_out = BatchNormalization(axis=3)(ff_out)
    out = Add()([out, ff_out])
    out = Activation('elu')(out)
    return out

def res_blk(ID, x, filter_depth, blk_depth):
    stride = (1,1) if ID == 1 else (2,2)
    x = conv_blk(x, ID * filter_depth, stride)
    for i in range(blk_depth - 1):
        x = ident_blk(x, ID * filter_depth)
    return x
    

def ResNet_50(in_shape):
    filter_depth = 64

    input = Input(in_shape)
    x = ZeroPadding2D((3,3))(input)
    x = Conv2D(64,
               kernel_size=(7,7),
               kernel_initializer='he_normal',
               strides=(2,2),
               padding='same')(x)

    x = BatchNormalization(axis=3)(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(3,3),
                     strides=(2,2),
                     padding='same')(x)

    res_depths = [3, 4, 6, 3]
    for i in range(len(res_depths)):
        x = res_blk(i + 1, x, (i + 1)*filter_depth, res_depths[i])
    
    x = AveragePooling2D(padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation=tf.nn.leaky_relu)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=x, name='ResNet-50')

    return model
    

