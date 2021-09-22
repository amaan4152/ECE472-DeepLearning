import struct as st
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import cv2

SHUFFLE_SIZE = 1000
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8

dataset = {
    'train-images': "../datasets/MNIST_DATASET/train-images-idx3-ubyte",
    'train-labels': "../datasets/MNIST_DATASET/train-labels-idx1-ubyte",
    'test-images': "../datasets/MNIST_DATASET/t10k-images-idx3-ubyte",
    'test-labels': "../datasets/MNIST_DATASET/t10k-labels-idx1-ubyte",
}


def parseDataset():
    images_set = []
    labels_set = []
    for k, v in dataset.items():
        if 'images' in k:
            with open(v, 'rb') as f: 
                magic, num_imgs, num_rows, num_cols = st.unpack('>IIII', f.read(16))
                images_set.append(np.fromfile(f, dtype=np.dtype(np.ubyte)).newbyteorder('>').reshape(num_imgs, num_rows, num_cols, 1))
        elif 'labels' in k: 
            with open(v, 'rb') as f: 
                magic, num_items = st.unpack('>II', f.read(8))
                labels_set.append(np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder('>'))

    return ((images_set[0], labels_set[0]), (images_set[1], labels_set[1]))

def dataset_config(ds):
    ds = ds.shuffle(SHUFFLE_SIZE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def data_gen(train, test):
    normalized_train, normalized_test = (train[0]/255., train[1]), (test[0]/255., test[1])
    train_dataset = tf.data.Dataset.from_tensor_slices(normalized_train)
    test_dataset = tf.data.Dataset.from_tensor_slices(normalized_test)
    train_dataset = dataset_config(train_dataset)
    return train_dataset, test_dataset

def main():

    # load data:
    # [0] => IMAGES
    # [1] => LABELS
    train, test = parseDataset()
    train_ds, test_ds = data_gen(train, test)

    model = models.Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=(train[0][0].shape[0], train[0][0].shape[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=train[0], y=train[1], batch_size=BATCH_SIZE, epochs=10, steps_per_epoch=np.math.ceil(0.8*train[0].shape[0]/BATCH_SIZE), validation_split=0.2)
    model.evaluate(test_ds)
    

if __name__ == "__main__":
    main()
