import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from layers.convolution import Convolution
from layers.max_pool import MaxPool
from layers.dense import Dense
from network import Network


def cross_entropy_loss(predictions, targets):
    num_samples = 10

    # avoid numerical instability by adding a small epsilon value
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss


def cross_entropy_loss_prime(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient


# instantiate the network wil the relevant layers
network = Network(cross_entropy_loss, cross_entropy_loss_prime)
network.add(Convolution((28, 28), 3, 1))
network.add(MaxPool(2))
network.add(Dense(169, 10))

# extract the training data from keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# setup training data
X_train = train_images[:5000] / 255
y_train = train_labels[:5000]

X_test = train_images[5000:10000] / 255
y_test = train_labels[5000:10000]

# convert to on-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# train network
network.train(X_train, y_train, 0.01, 200)

