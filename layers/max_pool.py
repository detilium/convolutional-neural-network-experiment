import numpy as np


class MaxPool:
    """
    The max pooling layer basically "shrinks" the input.
    For example: If the input data has a size og 26x26, and we've set out pool size to be 2, and our stride to be 2
    the output size would be 13x13, as the image "shrinks" by a factor of 2. Had we instead chosen a pool size of 3
    and a stride of 3, the image would shrink by a factor of 3 (≈8x8).

    The reasoning for this layer, is to reduce the number of parameters that needs to be optimized og updated
    during the back propagation, as well reduce the computational load of the network, without negating
    important details of the image to a certain degree.

    There are other types of pooling that can be used, such as "average pooling", but max pooling is vastly more used
    in modern convolutional neural networks.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        """
        Forward propagate through this layer
        :param input_data: Input data from the previous layer
        :return: Output
        """
        self.input_data = input_data

        # extract the number of channels (1 for grayscale, 3 for RGB),
        # the height of the input data, and the width of the input data (for example; 15x15)
        self.num_channels, self.input_height, self.input_width = input_data.shape

        # calculate the width and height of the output, based on the pool size
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # determine the output shape and instantiate the matrix
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        # loop through all channels
        for c in range(self.num_channels):
            # loop through the height of the output
            for i in range(self.output_height):
                # loop through the width of the output
                for j in range(self.output_width):
                    # calculate starting positions
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    # calculate end positions
                    end_i = start_i * self.pool_size
                    end_j = start_j * self.pool_size

                    # create a patch/window from the input data
                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    # find the maximum value from the patch/window
                    self.output[c, i, j] = np.max(patch)

        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backward propagate

        As there are no parameters to update in a max pooling layer, we simply transmit the maximum gradients obtained
        from the previous layer directly to the corresponding locations in the next layer.
        This process ensures that the maximum gradient values flow through the MaxPooling layer and continue
        propagating through the network.
        :param output_error: Gradient of the layer with respect to the output of this layer (dE/dY)
        :param learning_rate: Rate of learning
        :return: Gradient of the error with respect to the input (dE/dX)
        """
        input_error = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i * self.pool_size
                    end_j = start_j * self.pool_size

                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    # the mask is used to determine which values to propagate and which to discard.
                    # Only the maximum value within each patch/window is selected and passed to the previous layer,
                    # while the remaining values are set to 0.
                    mask = patch == np.max(patch)

                    input_error[c, start_i:end_i, start_j:end_j] = output_error[c, i, j] * mask

        return input_error
