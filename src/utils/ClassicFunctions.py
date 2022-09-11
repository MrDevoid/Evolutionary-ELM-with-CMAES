import numpy as np

class ClassicFunctions:
    def sigmoid(x):
        sig = 1 / (1 + np.exp(-x))
        return sig

    def swish(x):
        swi = np.multiply(x, 1 / (1 + np.exp(-x)))
        return swi
    
    def relu(x):
        rel = np.maximum(x,0)
        return rel

    def tanh(x):
        tan = np.tanh(x)
        return tan