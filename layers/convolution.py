import numpy as np
from scipy.signal import correlate2d


class Convolution:
    """
    A convolution is a dot product between the filter and the values of filter size at any given pixel on the image.
    It is called a convolution, as the operating is called "convolving" of two matrices.
        Example: If the first set of pixels in a filter size of (3, 3) in an image is:
            i = 1, j = 1
                [[0.0, 0.0, 0.0],
                [0.4, 0.5, 0.9],
                [1.0, 1.0, 1.0]]
        and the filter has the following random values:
                [[0.979], [0.278], [0.940],
                [0.713], [0.048], [0.564],
                [0.604], [0.327], [0.853]]
        the dot product of the two will be = 1.18, which will be stored at (i, j) = (1, 1) in the output matrix.

    The output shape of the convolution layer, will always be the same as the input, unless using padding.

    Filters is a convolution layer, can be though of as pattern detectors. Some filters in the early stages of a
    convolutional network, will be able to detect edges, some detect circles, some detect, corners, squares, etc.
    The deeper you delve into the neural network,the more complex patterns a filter will detect.
    Some filter will end up being animal detectors, detecting dogs, cats, mice, horses, etc.
    """
    def __init__(self, input_shape, filter_size, num_filters):
        """
        Instantiate the convolution layer
        :param input_shape: Size of the image. For example: 28x28 image: ``(28, 28)``
        :type input_shape: tuple
        :param filter_size: Size of the filter/kernel (for 3x3, this value should be 3)
        :type filter_size: int
        :param num_filters: The number of filters/kernels we need to product (for a grayscale image, this value should be 1).
                            For an RGB image, this filter should be 3 (1 red, 1 green, 1 blue)
        :type num_filters: int
        """
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape

        # in convolutional neural networks, the filters can be related to weights
        # in a traditional feed forward neural network
        self.filter_shape = (num_filters, filter_size, filter_size)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)

        # instantiate the filters with random values as a start, these will be tuned accordingly
        # once we back propagate the loss (the error)
        self.filters = np.random.randn(*self.filter_shape)
        # TODO: Why aren't the biases part of the calcuation?
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_data):
        """
        Forward propagate through the layer
        :param input_data: Input data from the previous layer
        :return: Output
        """
        self.input_data = input_data

        # initialize the output matrix
        output = np.zeros(self.output_shape)

        # generate the output for each filter (weight)
        for i in range(self.num_filters):
            output[i] = correlate2d(self.input_data, self.filters[i], mode='same')

        # apply ReLU activation function
        output = np.maximum(output, 0)
        return output

    def backward(self, output_error, learning_rate):
        """
        Backward propagate to update parameters
        :param output_error: Gradient of the error with respect to the output of this layer (dE/dY)
        :param learning_rate: The rate of learning
        :return: Gradient of the error with respect to the input data (dE/dX)
        """
        # instantiate the error with respect to the input, which will work as the output_error in the previous layer
        input_error = np.zeros_like(self.input_data)

        # instantiate the error with respect to the filters (weights)
        filter_error = np.zeros_like(self.filters)

        # loop through all filters to calculate the loss of a given filter
        for i in range(self.num_filters):
            # calculate the loss of the filter (weight)
            filter_error = correlate2d(self.input_data, output_error[i], mode='same')

            # calculate gradient of loss with respect to the input data
            input_error += correlate2d(output_error[i], filter_error[i], mode='same')

        # update parameters according to the learning rate
        self.filters -= learning_rate * filter_error
        self.biases -= learning_rate * output_error

        # return the error with respect to the input, so this can be used to tune the previous layer
        return input_error
