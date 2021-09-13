from re import X
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---- Global Variables ----
NUM_SAMPLES = 500
NUM_ITR = 1000
SEED = 1618
SIGMA_NOISE = 0.1
ROT_NUM = 2

# class for generating data
class Data(object):
    def __init__(self, num_samples, sigma, id, attr):

        # spiral attributes
        theta = np.random.uniform(attr["min"], attr["max"], size=(num_samples))
        spiral = self.Spiral(attr["center"], attr["gap"], theta, 1)

        # generate data
        noise = sigma * np.random.normal(size=(num_samples))  # gaussian noise
        self.x = id * spiral.r * np.cos(theta) / 1.5 + noise  # arbitrary scaling factor
        self.y = id * spiral.r * np.sin(theta) + noise

        self.spiral = spiral._data((self.x, self.y))

    # https://en.wikipedia.org/wiki/Archimedean_spiral
    class Spiral(object):
        def __init__(self, a, b, theta, n):
            self.r = a + b * (theta ** (1 / n))

        def _data(self, xy_dat):
            self.data = xy_dat
            return self


class Layer(object):
    """
    width -> number of neurons in layer
    X_in -> input data to layer
    """

    def __init__(self, id, width, X_in):
        num_features, num_samples = X_in.shape
        self.X = X_in
        self.W = tf.Variable(
            tf.random.normal(shape=[num_samples, num_features]),
            name=("WEIGHTS_" + str(id)),
        )
        self.B = tf.Variable(tf.zeros(shape=[width, num_features]), name=("BIASES_" + str(id)))
        print(tf.shape(self.W))
        print(tf.shape(self.X))
        print(tf.shape(self.B))
        exit(1)

    def ReLU(self, z):
        if z > 0: 
            return z
        else:
            return 0

    def _output(self):
        Z = self.W @ self.X + self.B
        return tf.vectorized_map(self.ReLU, Z.reshape(Z, shape=[Z.size]))


def main():
    np.random.seed(SEED)
    # generate 2 Archimidean spirals
    dataset = (
        Data(
            NUM_SAMPLES,
            SIGMA_NOISE,
            1,
            {"min": -ROT_NUM * 2 * np.pi + 0.1, "max": -0.1, "center": -1, "gap": 1},
        ),
        Data(
            NUM_SAMPLES,
            SIGMA_NOISE,
            -1,
            {"min": -ROT_NUM * 2 * np.pi + 0.1, "max": -0.1, "center": -1, "gap": 1},
        ),
    )

    input_data = np.concatenate([dataset[0].spiral.data, dataset[1].spiral.data], axis=1)
    layer_00 = Layer(0, 3, input_data)
    print(layer_00._output())


if __name__ == "__main__":
    main()
