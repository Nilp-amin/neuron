import numpy as np

from nn import NeuralNetwork
from layers import FullyConnectedLayer, ActiviationLayer
from activiations import tanh, tanh_prime
from costs import mse, mse_prime

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