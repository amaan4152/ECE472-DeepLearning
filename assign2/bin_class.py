import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import trange

# ---- Global Variables ----
NUM_SAMPLES = 10000
BATCH_SIZE = 16
NUM_ITR = 1000
LEARNING_RATE = 0.2
SEED = 1618
ALPHA = 0.01  # for leaky ReLU
SIGMA_NOISE = 0.1
ROT_NUM = 2

# class for generating data
class Data(object):
    def __init__(self, num_samples, sigma, id, attr):

        # spiral attributes
        theta = np.random.uniform(attr["min"], attr["max"], size=(num_samples))
        spiral = self.Spiral(attr["center"], attr["gap"], theta, 1)

        # generate data
        factor = 1 if id == 1 else -1
        noise = sigma * np.random.normal(size=(num_samples))  # gaussian noise
        self.x = (
            factor * spiral.r * np.cos(theta) / 1.5 + noise
        )  # arbitrary scaling factor
        self.y = factor * spiral.r * np.sin(theta) + noise

        self.spiral = spiral._data((self.x, self.y, [id] * num_samples))

    def _init_input(self, data):
        self.input_data = tf.constant(data, dtype=np.float32)

    def _batchGet(self, batch_size):
        self.index = NUM_SAMPLES * 2
        rand_ind = np.random.choice(self.index, size=batch_size)
        dat = tf.squeeze(tf.gather(self.input_data, rand_ind, axis=0))
        in_dat = tf.gather(dat, range(0, dat.shape[1] - 1), axis=1)
        in_labels = tf.gather(dat, dat.shape[1] - 1, axis=1)

        # normalize data
        return (
            (in_dat - tf.reduce_mean(in_dat)) / (tf.math.reduce_std(in_dat) ** 2),
            in_labels,
        )

    # https://en.wikipedia.org/wiki/Archimedean_spiral
    class Spiral(object):
        def __init__(self, a, b, theta, n):
            self.r = a + b * (theta ** (1 / n))

        def _data(self, xy_dat):
            self.data = xy_dat
            return self


class Layer(tf.Module):
    """
    width -> number of neurons in layer
    X_in -> input data to layer
    """

    def __init__(self, id, width):
        self.id = id
        self.width = width

    def _sigmoid(self, z):
        z = np.clip(
            z, -500, 500
        )  # https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
        return 1 / (1 + np.math.exp(-1 * z))

    def _ReLU(self, z):
        return max(0, z)

    def _leaky_ReLU(self, z):
        return max(ALPHA * z, z)

    def _batchNorm(self, input):
        mu = tf.reduce_mean(input)
        std = tf.math.reduce_std(input)
        return (input - mu) / std ** 2

    def _input(self, in_dat):
        self.X = in_dat
        self.num_features, self.num_samples = self.X.shape
        self.GAMMA = tf.Variable(
            tf.random.normal(shape=[self.width, 1]),
            name=("GAMMA_" + str(self.id)),
            dtype=np.float32,
        )
        self.W = tf.Variable(
            tf.random.normal(shape=[self.num_features, self.width]),
            name=("WEIGHTS_" + str(self.id)),
            dtype=np.float32,
        )

        self.BETA = tf.Variable(
            tf.zeros(shape=[self.width, 1]),
            name=("BETA_" + str(self.id)),
            dtype=np.float32,
        )
        self.activation_func = {
            "SIGMOID": self._sigmoid,
            "LRELU": self._leaky_ReLU,
            "RELU": self._ReLU,
        }

    def __call__(self, func):  # output from current layer
        self.Z = tf.squeeze((tf.transpose(self.W) @ self.X))
        self.Z = tf.squeeze(self.GAMMA * self._batchNorm(self.Z) + self.BETA)
        dims = self.Z.shape
        self.Z = tf.map_fn(
            self.activation_func[func], tf.reshape(self.Z, shape=[tf.size(self.Z)])
        )
        return tf.squeeze(tf.reshape(self.Z, shape=dims))


class MLP(object):
    def __init__(self, data, depth=1, width_array=[1]):
        self.depth = depth
        self.width_array = width_array
        self.data = data
        self.NN = []

        self.train_vars = []
        # initialize multiperceptron model
        for l in range(0, self.depth):
            layer = Layer(l, self.width_array[l])
            self.NN.append(layer)
            self.train_vars.append(layer.trainable_variables)
        self.NN.append(Layer(-1, 1))  # output layer

    def _loss_BCE(self, y, N):
        return tf.reduce_sum(
            y * tf.math.log(self.probs) + (1 - y) * tf.math.log(1 - self.probs)
        ) / (-1 * N)

    def _train(self):
        optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.99)
        bar = trange(NUM_ITR)
        for itr in bar:
            X, X_labels = self.data._batchGet(BATCH_SIZE)
            X = tf.transpose(X)
            with tf.GradientTape() as tape:
                for layer in self.NN:
                    layer._input(X)
                    if layer.id == -1:  # for output layer
                        self.probs = layer("SIGMOID")
                    else:
                        X = layer("RELU")

                loss = self._loss_BCE(X_labels, X_labels.shape[0])
            for vars in self.train_vars[-1:]:
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

            bar.set_description(f"Loss @ {itr} => {loss.numpy():0.6f}")
            bar.refresh()

    def _classify(self):
        layer_input = tf.gather(
            self.data.input_data, range(0, self.data.input_data.shape[1]), axis=1
        )
        for layer in self.NN:
            layer._input(layer_input)
            if layer.id == -1:
                probs = layer("SIGMOID")
            else:
                layer_input = layer("RELU")

        return probs

    # https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/
    def _decision_plot(self):
        features = tf.gather(self.data.input_data, [0, 1], axis=1).numpy()
        min1, max1 = features[:][0].min() - 1, features[:][0].max() + 1
        min2, max2 = features[:][1].min() - 1, features[:][1].max() + 1
        print(features[0])


# create a more generalized data object that takes in spiral objects rather than
# generating 2 seperate data objects
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
            0,
            {"min": -ROT_NUM * 2 * np.pi + 0.1, "max": -0.1, "center": -1, "gap": 1},
        ),
    )

    spiral_A = list(
        zip(
            dataset[0].spiral.data[0],
            dataset[0].spiral.data[1],
            dataset[0].spiral.data[2],
        )
    )
    spiral_B = list(
        zip(
            dataset[1].spiral.data[0],
            dataset[1].spiral.data[1],
            dataset[1].spiral.data[2],
        )
    )
    input_data = np.concatenate((spiral_A, spiral_B), axis=0)
    dataset[0]._init_input(input_data)
    mlp_model = MLP(dataset[0], 5, [32, 32, 16, 8, 4])
    mlp_model._train()
    mlp_model._classify()


if __name__ == "__main__":
    main()
