from __future__ import print_function
import math
import numpy as np
from elm import ElmModel
import matplotlib.pyplot as plt
from CMAESUtilities.CMAES import CMAES

class CMAESRegMultipleGaus(CMAES):
    def __init__(self, fitness_function, size_hidden_layer=None, elm=None, X_train=None, y_train=None, X_test=None, y_test=None, input_size=None, seed = None, problem_type = "CLASSIFICATION", gaussians_per_neuron = 1):
        super().__init__(fitness_function, size_hidden_layer, elm, X_train, y_train, X_test, y_test, input_size, seed, problem_type)
        self.__gaussians_per_neuron = gaussians_per_neuron

    def __get_number_of_element_per_neuron(self):
        return self.__gaussians_per_neuron*3


    def elm_function(self, element):
        reg = element[0]
        element = element[1:]
        def auxiliar(x): 
                sum = np.empty((0,x.shape[1]))
                elements_per_neuron = self.__get_number_of_element_per_neuron()
                limit = (len(x)//(len(element)//elements_per_neuron))
                for i in range(0, len(element)//elements_per_neuron):
                    aux = 0 
                    for j in range(0, self.__gaussians_per_neuron):                    
                        aux += element[(i+j)*elements_per_neuron]*math.e**(-element[(i+j)*elements_per_neuron+1]*(x[i*limit:(i+1)*limit]-element[(i+j)*elements_per_neuron+2])**2)
                    sum = np.append(sum, aux, axis=0)
                return sum
        self._elm_model.set_activation_function(auxiliar)
        self._elm_model.fit(self._x_train,self._y_train, reg=reg)
        output = self._elm_model.predict(self._x_test)
        if output.ndim ==1:
            true_positive = np.sum(np.where(output>0.5,1,0)==self._y_test)
        else:
            true_positive = np.sum(np.argmax(output, axis=1)==self._y_test)
        return (len(output) - true_positive)/len(output)