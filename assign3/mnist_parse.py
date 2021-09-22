# https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1

import struct as st
import numpy as np 

class Parser(object):
    def __init__(self, dataset):
        self.dataset = {}
        for k, v in dataset: 
            self.dataset[k] = open(v, 'rb')

            