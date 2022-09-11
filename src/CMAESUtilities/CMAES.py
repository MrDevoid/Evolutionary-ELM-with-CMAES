from __future__ import print_function
from tabnanny import check
import cma
import numpy as np
from elm import ElmModel
import utils.Logger as Logger

class CMAES:
    def __init__(self, fitness_function, size_hidden_layer=None, elm=None, X_train=None, y_train=None, X_test=None, y_test=None, input_size=None, seed = None, problem_type = "CLASSFICATION"):
        self.__fitness_function = fitness_function
        self.__size_hidden_layer = size_hidden_layer
        self.__seed = seed
        if self.__fitness_function == 'elm':
            if not elm is None:
                self._elm_model = elm
            elif input_size is None:
                self.__define_elm_model(len(X_train[0]))
            else:
                self.__define_elm_model(input_size)
        self._x_train = X_train
        self._y_train = y_train
        self._x_test = X_test
        self._y_test = y_test
        self._problem_type = problem_type

    def _get_output(self, output):
        if 'CLASSIFICATION' in self._problem_type:
            if output.ndim ==1:
                true_positive = np.sum(np.where(output>0.5,1,0)==self._y_test)
            else:
                true_positive = np.sum(np.argmax(output, axis=1)==self._y_test)
            return (len(output) - true_positive)/len(output)
        else:
            return np.sqrt(np.sum(np.subtract(output, self._y_test)**2) / len(output))

    def __define_elm_model(self, input_size):
        self._elm_model = ElmModel(input_size, self.__size_hidden_layer)

    def elm_function(self, element):
        pass

    def fitness(self, element):
        if self.__fitness_function == "elm":
            return self.elm_function(element)

    def generate_random_individual(self, dimensions, options=None):
        if self.__seed is None:
            random_individual = np.random.rand(dimensions)
        else:
            random_individual = self.__seed.rand(dimensions)
        if options is not None and "bounds" in options.keys():
            bounds = options["bounds"]
            for pos in range(0,len(bounds[0])):
                random_individual[pos] = random_individual[pos] * (bounds[1][pos]-bounds[0][pos]) + bounds[0][pos]
        return random_individual

    def problem_test(self, initial_value, initial_sigma, hidden_layer, iteration, opt=None, check_conditions = True):
        value = initial_value
        if isinstance(initial_value, int):
            initial_value = initial_value
            value = self.generate_random_individual(initial_value, opt)
        problem = cma.CMAEvolutionStrategy(value, initial_sigma, opt)
        best_individuals = np.empty((0,len(value)), dtype=object)
        worst_individuals = np.empty((0,len(value)), dtype=object)
        best_fitness = np.array([], dtype=np.float64)
        worst_fitness = np.array([], dtype=np.float64)
        mean_fitness = np.array([], dtype=np.float64)
        std_fitness = np.array([], dtype=np.float64)
        best_previous_population_fitness = None
        best_of_previous_population = None
        max_it = problem.opts['maxiter']

        numberIt = 0
        while not problem.stop(check_conditions) and (check_conditions or numberIt < max_it ):
            individuals = problem.ask()
            fitness_values = [self.fitness(indiv) for indiv in individuals]

            best_of_population = individuals[np.argmin(fitness_values)]
            best_population_fitness = np.min(fitness_values)

            if not best_previous_population_fitness is None and best_previous_population_fitness < best_population_fitness:
                individuals[np.argmax(fitness_values)] = best_of_previous_population
                fitness_values[np.argmax(fitness_values)] = best_previous_population_fitness
                best_of_population = best_of_previous_population
                best_population_fitness = best_previous_population_fitness

            best_individuals = np.append(best_individuals, np.array([individuals[np.argmin(fitness_values)]]), axis=0)
            worst_individuals = np.append(worst_individuals, np.array([individuals[np.argmax(fitness_values)]]), axis=0)
            best_fitness = np.append(best_fitness, np.min(fitness_values))
            worst_fitness = np.append(worst_fitness, np.max(fitness_values))
            mean_fitness = np.append(mean_fitness, np.mean(fitness_values))
            std_fitness = np.append(std_fitness, np.std(fitness_values))
            problem.tell(individuals, fitness_values)

            best_of_previous_population = best_of_population
            best_previous_population_fitness = best_population_fitness
            numberIt=numberIt+1
        Logger.log("Total iterations: "+str(numberIt))
        return problem, best_individuals, best_fitness, worst_individuals, worst_fitness, mean_fitness, std_fitness