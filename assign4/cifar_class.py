import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from matplotlib import pyplot as plt
from darse import Parser
from tensorflow.keras import models, regularizers
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten, MaxPooling2D, BatchNormalization

BATCH_SIZE = 128
EPOCHS = 40

dataset = {
    "train-01": "../datasets/CIFAR10_DATASET/pkl/data_batch_1",
    "train-02": "../datasets/CIFAR10_DATASET/pkl/data_batch_2",
    "train-03": "../datasets/CIFAR10_DATASET/pkl/data_batch_3",
    "train-04": "../datasets/CIFAR10_DATASET/pkl/data_batch_4",
    "train-05": "../datasets/CIFAR10_DATASET/pkl/data_batch_5",
    "test": "../datasets/CIFAR10_DATASET/pkl/test_batch",
}

def main():
    cifar_parser = Parser(dataset, 'CIFAR')
    train, test = cifar_parser.parse()
    train_data, train_labels = train
    test_data, test_labels = test

    # model
    model = models.Sequential()
    model.add(Conv2D(64,
            kernel_size=(3, 3),
            activation=tf.nn.elu,
            input_shape=(train_data.shape[1], train_data.shape[2], 3)))
    model.add(Conv2D(64,
            kernel_size=(3, 3),
            activation=tf.nn.elu))
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(Conv2D(256,
            kernel_size=(3, 3),
            activation=tf.nn.elu))
    model.add(Conv2D(256,
            kernel_size=(3, 3),
            activation=tf.nn.elu))
    model.add(Conv2D(256,
            kernel_size=(3, 3),
            activation=tf.nn.elu))
    model.add(Conv2D(256,
            kernel_size=(3, 3),
            activation=tf.nn.elu))
    model.add(Conv2D(256,
            kernel_size=(3, 3),
            activation=tf.nn.elu))
    model.add(MaxPooling2D(strides=(3,3)))
    model.add(Dropout(0.5))
    model.add(Dense(2048, 
              activation=tf.nn.leaky_relu,                      
              kernel_regularizer=regularizers.l2(l2=0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, 
              activation=tf.nn.leaky_relu,                      
              kernel_regularizer=regularizers.l2(l2=0.0005)))
    model.add(Dense(10, 
              activation='softmax',                      
              kernel_regularizer=regularizers.l2(l2=0.0005)))
    model.summary()

    # model compile
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x=train_data,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        steps_per_epoch=np.math.ceil(0.8 * train_data.shape[0] / BATCH_SIZE),
        validation_split=0.2,
    )
    model.evaluate(x=test_data, y=test_labels)
if __name__ == '__main__':
    main()
