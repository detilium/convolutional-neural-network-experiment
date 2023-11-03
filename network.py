import numpy as np


class Network:
    def __init__(self, loss_function, loss_function_prime):
        self.layers = []
        self.loss_function = loss_function
        self.loss_function_prime = loss_function_prime

    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0

            for i in range(len(X)):
                output = X[i]
                for layer in self.layers:
                    output = layer.forward(output)

                loss = self.loss_function(output.flatten(), y[i])
                total_loss += loss

                # convert to one-hot encoding
                one_hot_prediction = np.zeros_like(output)
                one_hot_prediction[np.argmax(output)] = 1
                one_hot_prediction = one_hot_prediction.flatten()

                num_prediction = np.argmax(one_hot_prediction)
                num_y = np.argmax(y[i])

                if num_prediction == num_y:
                    correct_predictions += 1

                # backwards propagation
                output_error = self.loss_function_prime(y[i], output.flatten()).reshape(-1, 1)

                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error, learning_rate)

            # epoch statistics
            average_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X) * 100.0
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

    def predict(self, input_sample):
        output = input_sample
        for layer in self.layers:
            output = layer.forward(output)

        return output
