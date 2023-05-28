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

class NeuralNetwork(object):
    def __init__(self) -> None:
        # user must define NN architecture
        # user must define cost function -> MSE, Cross entropy, ...
        # user must define optimizer -> Adam, SGD, ...
        # initially support Cross-entropy + SGD -> classification problems only
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

# activiation functions --------------------------

def sigmoid(data: np.array) -> np.array:
    return 1 / (1 + np.exp(-data))

def sigmoid_prime(data: np.array) -> np.array:
    return sigmoid(data) * (1 - sigmoid(data))

def softmax(data: np.array) -> np.array:
    return np.exp(data) / (np.sum(np.exp(data)))

def softmax_prime(data: np.array) -> np.array: # TODO: this outputs a matrix instead of a vector -> will this work?
    data = data.reshape(-1, 1)
    return np.diagflat(data) - data @ data.T

def tanh(data: np.array) -> np.array:
    return np.tanh(data)

def tanh_prime(data: np.array) -> np.array:
    return 1 - np.square(np.tanh(data))

# cost functions --------------------------------

def mse(guess: np.array, actual: np.array) -> np.array:
    return np.square(actual - guess).mean()

def mse_prime(guess: np.array, actual: np.array) -> np.array:
    return 2 * (guess - actual) / actual.size

def cross_entropy(guess: np.array, actual: np.array) -> np.array:
    return -np.sum(actual * np.log(guess))

def cross_entropy_prime(guess: np.array, actual: np.array) -> np.array:
    return -actual / guess

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    nn = NeuralNetwork()
    nn.add(FullyConnectedLayer(2, 3))
    nn.add(ActiviationLayer(tanh, tanh_prime))
    nn.add(FullyConnectedLayer(3, 1))
    nn.add(ActiviationLayer(tanh, tanh_prime))
    
    # train
    # training data
    input_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    output_train = np.array([[0], [1], [1], [0]])

    nn.use(mse, mse_prime)
    nn.train(input_train, output_train, ephocs=1000, learning_rate=0.1)

    # predict
    guess = nn.predict(np.array([1, 0]))
    print(guess)