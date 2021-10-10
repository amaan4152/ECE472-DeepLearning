from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from resnet import ResNet_N
from darse import Parser
from os import getpid
import numpy as np

CIFAR_TYPE = 100
BATCH_SIZE = 128
EPOCHS = 500

# CIFAR_10
dataset_10 = {
    "train-01": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_1",
    "train-02": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_2",
    "train-03": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_3",
    "train-04": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_4",
    "train-05": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/data_batch_5",
    "test": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR10_DATASET/pkl/test_batch",
}

# CIFAR_100
dataset_100 = {
    "train": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR100_DATASET/pkl/train",
    "test": "/zooper2/amaan.rahman/ECE472-DeepLearning/datasets/CIFAR100_DATASET/pkl/test",
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
	plt.tight_layout()
	# save plot to file
	plt.savefig('/zooper2/amaan.rahman/ECE472-DeepLearning/assign4/E1_cifar100_plot_' + str(getpid()) + '.png')
	plt.close()


def main():
	# dataset parse
	if CIFAR_TYPE == 10:
		dataset = dataset_10
	elif CIFAR_TYPE == 100:
		dataset = dataset_100

	cifar_parser = Parser(dataset, 'CIFAR_100')
	train, test = cifar_parser.parse()
	train_data, train_labels = train
	test_data, test_labels = test
	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)
	STEPS = 0.8 * train_data.shape[0] // BATCH_SIZE

	# model init
	model = ResNet_N(in_shape = (test_data.shape[1], test_data.shape[2], 3), 
					 layers = [3, 8, 8, 3], 
					 classes = 100) 
	model.summary()

	# model compile
	# https://towardsdatascience.com/super-convergence-with-cyclical-learning-rates-in-tensorflow-c1932b858252
	# https://arxiv.org/pdf/1506.01186.pdf
	model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=[TopKCategoricalAccuracy(), "accuracy"])

	# fit
	callback = ReduceLROnPlateau(monitor='val_loss', min_lr=1e-4, verbose=1)
	history = model.fit(
			x=train_data,
			y=train_labels,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			steps_per_epoch=STEPS,
			callbacks=[callback],
			validation_split=0.2
	)

	# evaluate
	model.evaluate(x=test_data, y=test_labels)
	plot_diagnostics(history)

if __name__ == '__main__':
    main()
