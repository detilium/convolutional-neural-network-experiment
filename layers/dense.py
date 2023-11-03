import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def softmax(self, z):
        # Shift the input values to avoid numerical instability
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)

        # Compute the softmax probabilities
        probabilities = exp_values / sum_exp_values

        return probabilities

    def softmax_prime(self, s):
        return np.diagflat(s) - np.dot(s, s.T)

    def forward(self, input_data):
        self.input_data = input_data

        # flatten matrix input from previous layer to a vector
        flattened_data = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_data.T) + self.biases

        self.output = self.softmax(self.z)

        return self.output

    def backward(self, output_error, learning_rate):
        # calculate the gradient of the loss with respect to the pre-activation (z) (dE/dY)
        output_error_z = np.dot(self.softmax_prime(self.output), output_error)

        # calculate the gradient of the loss with respect to the weights (dE/DW)
        weights_error = np.dot(output_error_z, self.input_data.flatten().reshape(1, -1))

        # calculate the gradient of the loss with respect to the biases (dE/dB)
        biases_error = output_error_z

        # calculate the gradient of the loss with respect to the input (dE/dX)
        input_error = np.dot(self.weights.T, output_error_z)
        input_error = input_error.reshape(self.input_data.shape)

        # update weights and biases based on the gradient and the learning rate
        self.weights -= weights_error * learning_rate
        self.weights -= biases_error * learning_rate

        # return the input error (dE/dY) to be used in the previous layer
        return input_error
