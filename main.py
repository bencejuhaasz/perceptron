import numpy as np
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * ( 1 - x)

training_inputs = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [1, 1, 1]])
                           #unknown sequence will be 1 0 1
training_outputs = np.array([[1, 0, 0]])
np.random.seed(1)
weights = 2*np.random.random((3, 1)).T-1

for i in range(20000):
    print("============================")
    output = sigmoid(np.dot(weights, training_inputs))
    print("outputs:")
    print(output)
    errors = training_outputs - output
    print("errors:")
    print(errors)
    weights+= errors*sigmoid_derivative(output)
    #time.sleep(1)

print("Training finished, testing unknown case:")
unkown_input=np.array([[1, 0, 1]]).T
print(unkown_input)
print(sigmoid(np.dot(weights, unkown_input)))
