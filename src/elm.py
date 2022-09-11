import numpy as np
from scipy.linalg import inv

class ElmModel:

    def __init__(self, input_size, hidden_layer_size=100, activation_function=None, seed = None):
        self.__input_size = input_size
        self.__hidden_layer_size = hidden_layer_size
        self.__activation_function = activation_function
        self.__seed = seed
        if activation_function is None:
            self.__activation_function = self.__relu
        self.__set_input_weights_with_biases()
        self.__output_weights = None

    def set_activation_function(self, activation_function):
        self.__activation_function = activation_function

    def __relu(self, x):
        return np.maximum(0, x)

    def __get_input_weights(self):
        if self.__seed is None:
            return np.random.uniform(low=-1, high=1, size=[self.__input_size, self.__hidden_layer_size])
        else:
            return self.__seed.uniform(low=-1, high=1, size=[self.__input_size, self.__hidden_layer_size])

    def __get_biases(self):
        if self.__seed is None:
            return np.random.uniform(low=-1, high=1, size=[self.__hidden_layer_size])
        else:
            return self.__seed.uniform(low=-1, high=1, size=[self.__hidden_layer_size])

    def __set_input_weights_with_biases(self):
        input_weights = self.__get_input_weights()
        biases = self.__get_biases()
        self.__input_weights_with_biases = np.vstack([biases, input_weights])

    def __get_moore_penrose_generalized_inverse(self, H, C):
        if H.shape[0] < H.shape[1]:
            regularization_matrix = np.identity(H.shape[0])*(1/C)
            return np.matmul(H.T, inv(np.add(regularization_matrix, np.matmul(H,H.T))))
        else:
            regularization_matrix = np.identity(H.shape[1])*(1/C)
            return np.matmul(inv(np.add(regularization_matrix, np.matmul(H.T,H))), H.T)

    def __get_output_weights(self, X, y, reg):
        H = self.__activation_function(np.dot(X,self.__input_weights_with_biases))
        moore_penrose = self.__get_moore_penrose_generalized_inverse(H, reg)
        output_weights = np.dot(moore_penrose, y)
        return output_weights

    def __prepare_X(self, X):
        return np.hstack([np.ones(X.shape[0]).reshape(X.shape[0],1),X])

    def __generate_elm_model(self, X, y, reg):
        X_with_ones = self.__prepare_X(X)
        self.__output_weights = self.__get_output_weights(X_with_ones, y, reg)

    def fit(self, X, y, reg=1):
        self.__generate_elm_model(X, y, reg)

    def predict(self, X):
        X_with_ones = self.__prepare_X(X)
        first_layer = self.__activation_function(np.dot(X_with_ones,self.__input_weights_with_biases))
        output = np.dot(first_layer, self.__output_weights)
        return output
    
    def print_weights(self):
        print(str(self.__input_weights_with_biases))

    def save(self,path):
        with open(path, 'w') as f:
            f.write("Input weights:\n")
            f.write(str(self.__input_weights_with_biases))
            f.write("\nOutput weights:\n")
            f.write(str(self.__output_weights))
            f.close()
    
    def read(self, path):
        with open(path, 'r') as f:
            file = f.read(path)
            

    def toString(self):
        s = ""
        s += "Input size: "+ str(self.__input_size) + "\n"
        s += "Hidden layer size: " + str(self.__hidden_layer_size) + "\n"
        s += "Input weights: " + str(self.__input_weights_with_biases) + "\n"
        s += "Output weights: " + str(self.__output_weights) + "\n"
        return s