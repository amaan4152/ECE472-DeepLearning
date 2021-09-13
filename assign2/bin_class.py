import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.python.ops.map_fn import map_fn

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
        self.width = width
        self.X = X_in
        self.num_input_neurons, num_samples, num_features = self.X.shape
        self.W = tf.Variable(
            tf.random.normal(shape=[width, num_samples]),
            name=("WEIGHTS_" + str(id)),
            dtype=np.float32,
        )
        self.B = tf.Variable(
            tf.zeros(shape=[width, num_features]),
            name=("BIASES_" + str(id)),
            dtype=np.float32,
        )

    def _sigmoid(self, z):
        return 1 / (1 + np.math.exp(-1 * z))

    def _output(self):
        Z = [
                tf.squeeze(
                    (tf.linalg.matvec(tf.transpose(self.X[i]), self.W) + self.B)
                )
                for i in range(0, self.num_input_neurons)
            ]
        Z = tf.stack(
            [
                tf.map_fn(
                    self._sigmoid,
                    Z[i][j],
                )
                for i in range(0, self.num_input_neurons)
                for j in range(0, self.width)
            ]
        )
        return Z


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

    spiral_A = list(zip(dataset[0].spiral.data[0], dataset[0].spiral.data[1]))
    spiral_B = list(zip(dataset[1].spiral.data[0], dataset[1].spiral.data[1]))
    input_data = tf.constant([spiral_A, spiral_B], dtype=np.float32)
    layer_00 = Layer(0, 3, input_data)
    print(layer_00._output())


if __name__ == "__main__":
    main()
