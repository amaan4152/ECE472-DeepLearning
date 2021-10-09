from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, Activation, Add, Input, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
def basic_blk(input, k, filter_depth, s):
    out = Conv2D(filter_depth,
                kernel_size=k[0],
                kernel_initializer='he_normal',  
                kernel_regularizer=regularizers.l2(l2=0.0001),
                padding='same',
                strides=s)(input)
                
    out = BatchNormalization(axis=3, momentum=0.9)(out)
    out = Activation('elu')(out)
    out = Conv2D(filter_depth,
                kernel_size=k[1],
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2=0.0001),
                padding='same')(out)

    out = BatchNormalization(axis=3, momentum=0.9)(out)
    return out

def ident_blk(input, filter_depth):
    ff_input = input
    out = basic_blk(input, (3,3), filter_depth, (1,1))
    out = Add()([out, ff_input])
    out = Activation('elu')(out)
    out = Dropout(0.5)(out)
    return out


def conv_blk(input, filter_depth, stride):
    ff_input = input
    out = basic_blk(input, (1,3), filter_depth, stride)
    ff_out = Conv2D(filter_depth,
                    kernel_size=(1,1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2=0.0001),
                    strides=stride,
                    padding='same')(ff_input)
    ff_out = BatchNormalization(axis=3)(ff_out)
    out = Add()([out, ff_out])
    out = Activation('elu')(out)
    out = Dropout(0.5)(out)
    return out


def res_blk(x, filter_depth, num_layers):
    x = conv_blk(x, filter_depth, (2,2))
    for i in range(num_layers - 1):
        x = ident_blk(x, filter_depth)
    return x


def ResNet_N(in_shape, N):
    filter_depth = 64
    input = Input(in_shape)

    # Preprocessing method: RANDOM CROP
    x = ZeroPadding2D(padding=(4,4))(input)
    x = RandomCrop(32, 32)(x)
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
        x = ident_blk(x, filter_depth)

    for i in range(len(layers[1:])):
        x = res_blk(x, (i + 2)*filter_depth, layers[i+1])
    
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(l2=0.00001))(x)
    model = Model(inputs=input, outputs=x, name=('ResNet-' + str(4*N + 2)))

    return model
    

