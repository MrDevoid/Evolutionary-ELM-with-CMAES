import numpy as np

class MnistScaler:
    def __init__(self, min_of_new_range= 0, max_of_new_range = 1):
        self.__min_of_new_range = min_of_new_range
        self.__max_of_new_range = max_of_new_range
        self.__max_value = 255
        self.__min_value = 0

    def fit(self, data):
        return self

    def transform(self, data):
        multiplier = (self.__max_of_new_range-self.__min_of_new_range)
        less_data = np.subtract(data,self.__min_value)
        multiplied_data = multiplier*less_data
        divided_data = np.divide(multiplied_data,self.__max_value)
        data = np.add(divided_data, self.__min_of_new_range)
        return data