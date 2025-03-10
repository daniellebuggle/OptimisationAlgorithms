import numpy as np


def generate_trainingdata(m=25):
    return np.array([0, 0]) + 0.25 * np.random.randn(m, 2)


def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y = 0;
    count = 0
    for w in minibatch:
        z = x - w - 1
        y = y + min(37 * (z[0] ** 2 + z[1] ** 2), (z[0] + 5) ** 2 + (z[1] + 5) ** 2)
        count = count + 1
    return y / count
