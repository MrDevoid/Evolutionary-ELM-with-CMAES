from __future__ import print_function
import math
import numpy as np
from elm import ElmModel
import matplotlib.pyplot as plt
from CMAESUtilities.CMAES import CMAES

class CMAESReg(CMAES):
    def __init__(self, fitness_function, size_hidden_layer=None, elm=None, X_train=None, y_train=None, X_test=None, y_test=None, input_size=None, seed = None, problem_type = "CLASSIFICATION"):
        super().__init__(fitness_function, size_hidden_layer, elm, X_train, y_train, X_test, y_test, input_size, seed, problem_type)

    def elm_function(self, element):
        reg = 10**element[0]
        gaussians = element[1:]
        def auxiliar(x):
                sum = 0
                for i in range(0, len(gaussians)//3):
                    sum += gaussians[i*3]*math.e**(-gaussians[i*3+1]*(x-gaussians[i*3+2])**2)
                return sum
        self._elm_model.set_activation_function(auxiliar)
        self._elm_model.fit(self._x_train,self._y_train, reg)
        output = self._elm_model.predict(self._x_test)
        return self._get_output(output)
        