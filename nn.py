import numpy as np

from typing import Callable
from layers import Layer

class NeuralNetwork(object):
    def __init__(self) -> None:
        self.layers = [] 
        self.cost = None
        self.cost_prime = None

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def use(self, cost: Callable, cost_prime: Callable) -> None:
        self.cost = cost
        self.cost_prime = cost_prime

    def train(self, train_inputs: np.array, train_outputs: np.array, ephocs: int, learning_rate: float) -> None:
        samples = train_inputs.shape[0]
        for ephoc in range(ephocs):
            cost = 0
            for i, _input in enumerate(train_inputs):
                # forward propagate 
                output = _input
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # for visual purposes only
                cost += self.cost(output, train_outputs[i])

                # calculate cost function 
                error = self.cost_prime(output, train_outputs[i]) 

                # back propagate
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            # calculate average cost across all samples
            cost /= samples
            print(f"{ephoc=}  {cost=}")

    def predict(self, data: np.array) -> np.array:
        output = data 
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output