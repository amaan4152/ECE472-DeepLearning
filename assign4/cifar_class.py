import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from darse import Parser

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
    plt.imshow(train[0][0])
    plt.savefig('img_00.png')

if __name__ == '__main__':
    main()
