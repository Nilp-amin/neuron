import csv
import numpy as np

seed = 0

class NeuralNetwork(object):
    # TODO: Fix naming of weights -> input layer weights -> hidden layer weights and hidden layer weights -> output layer weights
    def __init__(self) -> None:
        # user must define NN architecture
        # user must define cost function -> MSE, Cross entropy, ...
        # user must define optimizer -> Adam, SGD, ...
        # initially support Cross-entropy + SGD -> classification problems only

        np.random.seed(seed)

        # Input: 7 neurons
        self._input_layer = np.random.random((5, 7))
        # Hidden: 5 neurons -> sigmoid
        self._hidden_layer = np.random.random((3, 5)) 
        # Output: 3 neurons -> softmax

        # intermediate layer outputs for back propagation
        self._layer_1 = None
        self._layer_2 = None

    def train(self, path: str, learning_rate: float) -> None:
        # loading data
        # loop below until cost function is minimised/maximised:
        #   forward propagating
        #   backward propogating
        guess = self._forward_propogate(np.random.random(7))
        self._back_propogate(guess, np.array([1, 0, 0]), learning_rate)

    def _forward_propogate(self, data: np.array) -> np.array:
        assert data.shape[0] == self._input_layer.shape[1], "input data size does not match input layer size"

        # output of input layer
        layer_1 = self._input_layer @ data
        # output of hidden layer
        layer_2 = self._sigmoid(self._hidden_layer @ layer_1)
        # output of output layer
        output = self._softmax(layer_2)

        return output

    def _back_propogate(self, guess: np.array, actual: np.array, alpha: float) -> None:
        cost = self._cross_entropy(guess, actual)

        # update hidden layer weights
        output_gradient = np.matrix(guess - actual)
        self._hidden_layer -= alpha * np.transpose(output_gradient) 

        # TODO: update input layer weights
        hidden_gradient = ... 
        self._input_layer -= alpha * ...

    def _sigmoid(self, data: np.array) -> np.array:
        return 1 / (1 + np.exp(-data))

    def _softmax(self, data: np.array) -> np.array:
        return np.exp(data) / np.sum(np.exp(data))

    def _cross_entropy(self, guess: np.array, actual: np.array) -> np.array:
        return -np.sum(actual * np.log(guess))
    
    def _mse(self, guess: np.array, actual: np.array) -> np.array:
        return np.square(actual - guess).mean()

    def _load_csv(self, path: str) -> np.array:
        pass

if __name__ == "__main__":
    test_data_path = f""
    train_data_path = f""

    nn = NeuralNetwork()
    nn.train(train_data_path, 0.1)
    # nn.predict(test_data_path)