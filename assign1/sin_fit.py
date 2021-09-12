"""
Amaan Rahman
ECE 472: Deep Learning
Professor Curro

Note: This program took assistance from the provided example by Professor Curro; however all work was done independently by, Amaan Rahman. 
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

matplotlib.style.use("classic")
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("num_basis", 3, "Number of basis functions")  # M
flags.DEFINE_integer("batch_size", 15, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 2000, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.0901, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.1, "Standard deviation of noise random variable")


class Data(object):
    def __init__(self, num_samp, sigma, random_seed):

        np.random.seed(random_seed)
        # We’re going to learn these paramters
        self.index = np.arange(num_samp)
        self.x = np.random.uniform(0.1, 0.9, size=(num_samp, 1))
        self.y_noise = np.sin(2 * np.pi * self.x) + sigma * np.random.normal(
            size=(num_samp, 1)
        )  # with noise

        self.y = np.sin(2 * np.pi * self.x)  # no noise

    def get_batch(self, batch_size):

        choices = np.random.choice(self.index, size=batch_size)
        return (
            self.x[choices],
            self.y_noise[choices].flatten(),
            self.y[choices].flatten(),
        )


class Model(tf.Module):
    def __init__(self):

        self.w = tf.Variable(tf.random.normal(shape=[FLAGS.num_basis]), name="weights")
        self.b = tf.Variable(tf.zeros(shape=[1]), name="bias")
        self.mu = tf.Variable(tf.random.normal(shape=[FLAGS.num_basis]), name="mean")
        self.std = tf.Variable(tf.random.normal(shape=[FLAGS.num_basis]), name="stdev")

    def __call__(self, x):
        self.basis = [
            tf.squeeze(
                (tf.math.exp(-1 * (((x - self.mu[i]) ** 2) / (self.std[i] ** 2))))
                * self.w[i]
                + self.b
            )
            for i in range(0, FLAGS.num_basis)
        ]

        return tf.math.add_n(self.basis)


def train(data, model, withNoise):
    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    bar = trange(FLAGS.num_iters)

    for i in bar:
        with tf.GradientTape() as tape:
            x, y_noise, y_clean = data.get_batch(FLAGS.batch_size)
            if withNoise:
                y = y_noise
            else:
                y = y_clean
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()


def main(a):
    data = Data(
        FLAGS.num_samples,
        FLAGS.sigma_noise,
        FLAGS.random_seed,
    )
    FLAGS.num_basis = 3
    model_noiseless = Model()
    FLAGS.num_basis = 3
    model_noise = Model()

    train(data, model_noiseless, False)
    train(data, model_noise, True)

    fig, ax = plt.subplots(2, 2)
    xs = np.linspace(-0.2, 1, 40)
    ys = np.sin(2 * np.pi * xs)
    xs = xs[:, np.newaxis]
    ax[0, 0].plot(
        xs, np.squeeze(model_noiseless(xs)), "--", np.squeeze(data.x), data.y_noise, "o"
    )
    ax[0, 0].plot(xs, ys, "-")
    ax[0, 0].set_title("Fit 1 (No Noise)")
    ax[1, 0].plot(
        xs, np.squeeze(model_noise(xs)), "--", np.squeeze(data.x), data.y, "o"
    )
    ax[1, 0].plot(xs, ys, "-")
    ax[1, 0].set_title("Fit 2 (With Noise)")

    for basis in model_noiseless.basis:
        ax[0, 1].plot(np.squeeze(basis))
    for basis in model_noise.basis:
        ax[1, 1].plot(np.squeeze(basis))
    ax[0, 1].set_title("Basis for Fit 1")
    ax[1, 1].set_title("Basis for Fit 2")
    for axis in ax.flat:
        axis.set(xlabel="x", ylabel="y")
    plt.tight_layout()
    plt.savefig("fit.pdf")


if __name__ == "__main__":
    app.run(main)
