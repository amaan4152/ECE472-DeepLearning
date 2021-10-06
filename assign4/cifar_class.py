import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import LearningRateScheduler
from assign4.resnet import ResNet_50
from darse import Parser
from tensorflow.keras import models, regularizers
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten, MaxPooling2D, BatchNormalization, Activation, Add
from os import getpid

BATCH_SIZE = 64
EPOCHS = 40

dataset = {
    "train-01": "../datasets/CIFAR10_DATASET/pkl/data_batch_1",
    "train-02": "../datasets/CIFAR10_DATASET/pkl/data_batch_2",
    "train-03": "../datasets/CIFAR10_DATASET/pkl/data_batch_3",
    "train-04": "../datasets/CIFAR10_DATASET/pkl/data_batch_4",
    "train-05": "../datasets/CIFAR10_DATASET/pkl/data_batch_5",
    "test": "../datasets/CIFAR10_DATASET/pkl/test_batch",
}

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
def plot_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='validation')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='validation')
	plt.legend()
	# save plot to file
	plt.savefig('cifar10_plot_' + str(getpid()) + '.png')
	plt.close()

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
def lr_sched(epoch, lr):
    if epoch >= 5: 
            return lr * tf.math.exp(-0.1)
    return lr

# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
def VGG_blk(input, filter_depth):
	out = Conv2D(filter_depth,
				 kernel_size=(3,3),
				 kernel_initializer='he_normal',
				 padding='same')(input)

	out = BatchNormalization(axis=3)(out)
	out = Activation('elu')(out)
	out = Conv2D(filter_depth,
				 kernel_size=(3,3),
				 kernel_initializer='he_normal',
				 padding='same')(out)

	out = BatchNormalization(axis=3)(out)
	return out

def ident_blk(input, filter_depth):
	ff_input = input
	out = VGG_blk(input, filter_depth)
	out = Add()([out, ff_input])
	out = Activation('elu')(out)
	return out

def conv_blk(input, filter_depth):
	ff_input = input
	out = VGG_blk(input, filter_depth)
	ff_out = Conv2D(filter_depth,
					kernel_size=(1,1),
					kernel_initializer='he_normal',
					strides=(2,2))(ff_input)

	out = Add()([out, ff_out])
	out = Activation('elu')(out)
	return out

def main():
	cifar_parser = Parser(dataset, 'CIFAR')
	train, test = cifar_parser.parse()
	train_data, train_labels = train
	test_data, test_labels = test
	# model
	model = ResNet_50((test_data.shape[1], test_data.shape[2], 3))
	model.summary()
	# model compile
	model.compile(
			optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
	)
	callback = LearningRateScheduler(lr_sched)
	history = model.fit(
			x=train_data,
			y=train_labels,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			steps_per_epoch=np.math.ceil(0.8 * train_data.shape[0] / BATCH_SIZE),
			validation_split=0.2,
			callbacks=[callback]
	)
	model.evaluate(x=test_data, y=test_labels)
	plot_diagnostics(history)

if __name__ == '__main__':
    main()
