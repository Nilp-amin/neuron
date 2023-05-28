from typing import Callable

import numpy as np

class Layer(object):
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_propagation(self, data: np.array) -> np.array:
        raise NotImplementedError

    def back_propagation(self, output_error: np.array, alpha: float) -> np.array:
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.random((input_size, output_size))
        self.bias = np.random.random((1, output_size))

    def forward_propagation(self, data: np.array) -> np.array:
        self.input = data
        self.output = self.input @ self.weights + self.bias

        return self.output 

    def back_propagation(self, output_error: np.array, alpha: float) -> np.array:
        input_error = output_error @ self.weights.T
        weights_error = np.matrix(self.input).T @ output_error
        bias_error = output_error

        self.bias -= alpha * bias_error 
        self.weights -= alpha * weights_error

        return input_error 

class ActiviationLayer(Layer):
    def __init__(self, activation: Callable, activation_prime: Callable) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, data: np.array) -> np.array:
        self.input = data
        self.output = self.activation(self.input)

        return self.output 

    def back_propagation(self, output_error: np.array, alpha: float) -> np.array:
        return output_error * self.activation_prime(self.input)