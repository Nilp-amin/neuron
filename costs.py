import numpy as np

def mse(guess: np.array, actual: np.array) -> np.array:
    return np.square(actual - guess).mean()

def mse_prime(guess: np.array, actual: np.array) -> np.array:
    return 2 * (guess - actual) / actual.size

def cross_entropy(guess: np.array, actual: np.array) -> np.array:
    return -np.sum(actual * np.log(guess))

def cross_entropy_prime(guess: np.array, actual: np.array) -> np.array:
    return -actual / guess