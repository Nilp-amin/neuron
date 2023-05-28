import numpy as np

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