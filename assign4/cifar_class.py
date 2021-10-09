import tensorflow as tf
from tensorflow_addons.optimizers import CyclicalLearningRate
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from resnet import ResNet_N
from darse import Parser
from os import getpid

BATCH_SIZE = 128
EPOCHS = 500

dataset = {
    "train-01": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_1",
    "train-02": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_2",
    "train-03": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_3",
    "train-04": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_4",
    "train-05": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_5",
    "test": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/test_batch",
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
	plt.savefig('/zooper2/amaan.rahman/ECE472-DeepLearning/assign4/cifar10_plot_' + str(getpid()) + '.png')
	plt.close()

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
def lr_sched(epoch, lr):
	if epoch == 1:
		return lr * 10
	if epoch >= 10: 
		if epoch % 5 == 0:
			return lr/10
	return lr

def main():
	# dataset parse
	cifar_parser = Parser(dataset, 'CIFAR')
	train, test = cifar_parser.parse()
	train_data, train_labels = train
	test_data, test_labels = test

	# model init
	model = ResNet_N((test_data.shape[1], test_data.shape[2], 3), N=3) # N = 4 => ResNet_18
	model.summary()

	# model compile
	# https://towardsdatascience.com/super-convergence-with-cyclical-learning-rates-in-tensorflow-c1932b858252
	# https://arxiv.org/pdf/1506.01186.pdf
	STEPS = 0.8 * train_data.shape[0] // BATCH_SIZE
	CLR = CyclicalLearningRate(initial_learning_rate=5e-4, maximal_learning_rate=1e-1, step_size=2*STEPS, scale_fn=lambda x: 1 / (2.0 ** (x - 1)))
	model.compile(optimizer=Adam(learning_rate=CLR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	# fit
	#callback = ReduceLROnPlateau(monitor='val_loss', verbose=1)
	history = model.fit(
			x=train_data,
			y=train_labels,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			steps_per_epoch=STEPS,
			validation_split=0.2
	)

	# eval
	model.evaluate(x=test_data, y=test_labels)
	plot_diagnostics(history)

if __name__ == '__main__':
    main()
