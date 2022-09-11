from CMAESUtilities import CMAESFactory as CMAF
from elm import ElmModel
import utils.Loader as Ld
import utils.Printer as printer
import utils.Logger as Logger
import utils.CommonFuncs as ComFunc
import itertools as it
import matplotlib
import numpy as np
import pandas as pd
import math

def get_output(output, test, problem_type):
    if 'CLASSIFICATION' in problem_type:
        if output.ndim ==1:
            true_positive = np.sum(np.where(output>0.5,1,0)==test)
        else:
            true_positive = np.sum(np.argmax(output, axis=1)==test)
        return (len(output) - true_positive)/len(output)
    else:
        return np.sqrt(np.sum(np.subtract(output, test)**2) / len(output))

def get_bounds(problem_type, bounds_reg, bounds_w, bounds_g, bounds_c):
    cartesian_product = []
    if "REG" in problem_type:
        cartesian_product = list(it.product(bounds_reg, bounds_w, bounds_g, bounds_c))
    else:
        cartesian_product = list(it.product(bounds_w, bounds_g, bounds_c))
    list_of_bounds_limits = list(map(lambda x: list(map(list, list(zip(*x)))), cartesian_product))
    return list_of_bounds_limits

def get_gen_labels(type_exe, dimensions):
    if 'REG' in type_exe:
        dimensions = dimensions + 1
    return list(map(lambda x: 'gen'+str(x), range(0,dimensions)))

def test_elm(problem_type, data_to_test, hidden_layer_size, input_means, input_sigma, type_exe, options=None, elm=None, i=0, check_conditions=True, gaussians_per_neuron = 1, seed2=None):
    elm_problem = CMAF.create_cmaes_problem(type_exe, hidden_layer_size, elm, data_to_test, seed2, problem_type, gaussians_per_neuron)    
    problem, bests_individuals, bests_fitness, worsts_individuals, worsts_fitness, means_fitness, std_fitness = elm_problem.problem_test(input_means, input_sigma, hidden_layer_size,iteration=i, opt = options, check_conditions = check_conditions)
    return problem.result[0], problem.result[1], bests_individuals, bests_fitness, worsts_individuals, worsts_fitness, means_fitness, std_fitness

def get_middle_points(domain):
    domain_mins = domain[0]
    domain_maxs = domain[1]
    result = np.divide(np.add(domain_mins, domain_maxs),2)
    return result

def run_test(problem_type, numero_gaussianas, hidden_layer_size, initial_sigma, path, domain,seed, scaler=None, options=None, dataset="mnist", type_exe="REG", typea= "REG", graphic_itf=True, check_conditions = True, gaussians_per_neuron = 1, **args):
    data_to_test= Ld.get_data(problem_type, dataset, **args)
    if scaler is not None:
        scaler = scaler.fit(data_to_test['X_train'])
        data_to_test['X_train'] = scaler.transform(data_to_test['X_train'])
        data_to_test['X_test'] = scaler.transform(data_to_test['X_test'])
    auxiliar_options = options.copy()
    elm = ElmModel(len(data_to_test['X_train'][0]), hidden_layer_size, seed = seed)
    for dimensions in np.multiply(3,numero_gaussianas):
        seed_for_cmaes = np.random.RandomState(123)
        for i in range(0,5):
            auxiliar_dimensions = dimensions
            if "REG" in typea:
                auxiliar_dimensions = auxiliar_dimensions + 1
            Logger.log("Test with " + str(dimensions) + " dimensions:")
            if auxiliar_options is not None and 'bounds' in auxiliar_options.keys():
                if len(options['bounds'][0])%3 != 0:
                    reg_min = options['bounds'][0][0]
                    reg_max = options['bounds'][1][0]
                    rest = [options['bounds'][0][1:], options['bounds'][1][1:]] 
                    auxiliar_options['bounds'] = [[reg_min]+rest[0]*(auxiliar_dimensions//3), [reg_max]+rest[1]*(auxiliar_dimensions//3)]
                else:
                    auxiliar_options['bounds'] = [options['bounds'][0]*(auxiliar_dimensions//3), options['bounds'][1]*(auxiliar_dimensions//3)]
            
            # random means
            #sol, best, bests_individuals, bests_fitness, worsts_individuals, worsts_fitness, means_fitness, std_fitness = test_elm(data_to_test, hidden_layer_size, int(auxiliar_dimensions), initial_sigma, typea, auxiliar_options, elm, i, check_conditions, gaussians_per_neuron, seed2 = seed_for_cmaes)
            
            # fixed means
            sol, best, bests_individuals, bests_fitness, worsts_individuals, worsts_fitness, means_fitness, std_fitness = test_elm(problem_type, data_to_test, hidden_layer_size, get_middle_points(auxiliar_options['bounds']), initial_sigma, typea, auxiliar_options, elm, i, check_conditions, gaussians_per_neuron, seed2 = seed_for_cmaes)
            
            Logger.log("    Solution: " + str(sol))
            Logger.log("    Best: " + str(best))
            Logger.log("    Bests_individuals: " + str(bests_individuals))
            Logger.log("    Bests_fitness: " + str(bests_fitness))
            Logger.log("    Worsts_individuals: " + str(worsts_individuals))
            Logger.log("    Worsts_fitness: " + str(worsts_fitness))
            Logger.log("    Means_fitness: " + str(means_fitness))
            
            printer.plot_one_object(bests_individuals, "Mejores individuos", "Iteraciones", "Genes de los individuos", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= i, name_of_image= "best_individuals"), get_gen_labels(typea, dimensions), graphic_itf)
            printer.plot_one_object(bests_fitness, "Mejores valores de adaptacion", "Iteraciones", "Evolucion del mejor fitness", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= i, name_of_image= "bests_fitness"), None, graphic_itf)
            printer.plot_one_object(worsts_individuals, "Peores individuos", "Iteraciones", "Genes de los individuos", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= i, name_of_image= "worsts_individuals"), get_gen_labels(typea, dimensions), graphic_itf)
            printer.plot_one_object(worsts_fitness, "Peores valores de adaptacion", "Iteraciones", "Evolucion del peor fitness", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= i, name_of_image= "worsts_fitness"), None, graphic_itf)
            printer.plot_means_object(means_fitness, std_fitness, bests_fitness, "Media valores de adaptacion", "Iteraciones", "Evolucion de la media del fitness",path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= i, name_of_image= "means_fitness"), None, graphic_itf)
   
def gaussians(individual, val):
        sum = 0
        for i in range(0, len(individual)//3):
            sum += individual[i*3]*math.e**(-individual[i*3+1]*(val-individual[i*3+2])**2)
        return sum

def set_function_and_execute_elm(elm, activation_function, data_to_test, regularization = 1, problem_type = 'CLASSIFICATION'):
    elm.set_activation_function(activation_function)
    elm.fit(data_to_test['X_train'], data_to_test['y_train'], reg = regularization)
    output = elm.predict(data_to_test['X_test'])
    return get_output(output, data_to_test['y_test'], problem_type)

def test_reg_fixed_functions(problem_type, hidden_layer_size, classic_function, data, range_of_exponents, scaler):
    data_to_test  = Ld.get_data(problem_type, data)
    if scaler:
        scaler = scaler.fit(data_to_test['X_train'])
        data_to_test['X_train'] = scaler.transform(data_to_test['X_train'])
        data_to_test['X_test'] = scaler.transform(data_to_test['X_test'])
    elm = ElmModel(len(data_to_test['X_train'][0]), hidden_layer_size, seed=np.random.RandomState(1234))
    errors = list(map(lambda x: set_function_and_execute_elm(elm, classic_function, data_to_test, 10**x, problem_type), range_of_exponents))
    for error, exp in zip(errors, range_of_exponents):
        Logger.log("Exp= " + str(exp))
        Logger.log("Error with classic function: " + str(error))

def compare_trained_with_classics_activation_function(problem_type, hidden_layer_size, data, trained_functions, classic_functions, regs_for_trained, regs_for_classic, scaler, **args):
    data_to_test = Ld.get_data(problem_type, data, **args)
    if scaler:
        scaler = scaler.fit(data_to_test['X_train'])
        data_to_test['X_train'] = scaler.transform(data_to_test['X_train'])
        data_to_test['X_test'] = scaler.transform(data_to_test['X_test'])
    elm = ElmModel(len(data_to_test['X_train'][0]), hidden_layer_size, seed=np.random.RandomState(1234))
    results_trained_functions = list(map(lambda reg_value: list(map(lambda fun: set_function_and_execute_elm(elm, fun, data_to_test, regularization=10**reg_value, problem_type = problem_type), trained_functions)), regs_for_trained))
    results_classic_functions = list(map(lambda reg_value: list(map(lambda fun: set_function_and_execute_elm(elm, fun, data_to_test, regularization=10**reg_value, problem_type = problem_type), classic_functions)), regs_for_classic))
    for number, result in zip(range(0, len(results_trained_functions)), results_trained_functions):
        Logger.log("Error with trained function "+str(number)+": "+str(result))
    for number, result in zip(range(0, len(results_classic_functions)), results_classic_functions):
        Logger.log("Error with classic function "+str(number)+": "+str(result))

def print_several_activation_function_together(values, path, print_in_range=range(-10, 10)):
    matplotlib.use('Agg')
    printer.print_function(values, (lambda x, y: gaussians(x, y)), path, print_in_range)

def main_test(problem_type, type_exe, bounds_dict, pop_size, max_iter, sigmas, hidden_layer_sizes, number_of_gaussians, dataset, graphic_itf, check_conditions, gaus_per_neuron,scaler, path, **kwargs):
    matplotlib.use('Agg')
    typea = ComFunc.get_type(type_exe)
    counter=1
    bounds = get_bounds(typea, bounds_dict['bounds_reg'], bounds_dict['bounds_w'], bounds_dict['bounds_g'], bounds_dict['bounds_c'])
    for bound in bounds:
        options = {'bounds': bound, 'popsize_factor': pop_size, 'maxiter': max_iter}
        for sigma in sigmas:
            Logger.log("tests_with_sigma: " + str(sigma))
            for hidden_layer_size in hidden_layer_sizes:
                seed1 = np.random.RandomState(1234)
                try:
                    run_test(problem_type, number_of_gaussians, hidden_layer_size, sigma, path=path, domain=counter, seed=seed1, scaler=scaler, options=options, dataset=dataset, type_exe=type_exe, typea=typea, graphic_itf=graphic_itf, check_conditions = check_conditions, gaussians_per_neuron = gaus_per_neuron, **kwargs)
                except Exception as e:
                    Logger.log(e)
                    Logger.log("No se ha podido completar el evolutivo")
        counter = counter+1


def run_test_cv(problem_type, numero_gaussianas, hidden_layer_size, initial_sigma, path, domain,seed, scaler=None, options=None, dataset="mnist", type_exe="REG", typea= "REG", graphic_itf=True, check_conditions = True, gaussians_per_neuron = 1, **args):
    cv_dictionary_to_test= Ld.get_data_cv(problem_type, dataset, **args)
    if scaler is not None:
        scaler = scaler.fit(cv_dictionary_to_test['X_train'])
        cv_dictionary_to_test['X_train'] = scaler.transform(cv_dictionary_to_test['X_train'])
    auxiliar_options = options.copy()
    elm = ElmModel(len(cv_dictionary_to_test['X_train'][0]), hidden_layer_size, seed = seed)
    for dimensions in np.multiply(3,numero_gaussianas):
        for train_index, test_index in cv_dictionary_to_test['splits'].split(cv_dictionary_to_test['X_train']):
            data = Ld.prepare_data_cv(cv_dictionary_to_test, train_index, test_index, test_size = args.get('test_size'))
            data_to_train = {'X_train':data['X_train'],'y_train':data['y_train'],'X_test':data['X_validation'],'y_test':data['y_validation']}
            data_to_test = {'X_train': np.concatenate((data['X_train'], data['X_validation']), axis=0), 'y_train': np.concatenate((data['y_train'], data['y_validation']), axis=0), 'X_test':data['X_test'], 'y_test':data['y_test']}
            seed_for_cmaes = np.random.RandomState(123)
            auxiliar_dimensions = dimensions
            if "REG" in typea:
                auxiliar_dimensions = auxiliar_dimensions + 1
            Logger.log("Test with " + str(dimensions) + " dimensions:")
            if auxiliar_options is not None and 'bounds' in auxiliar_options.keys():
                if len(options['bounds'][0])%3 != 0:
                    reg_min = options['bounds'][0][0]
                    reg_max = options['bounds'][1][0]
                    rest = [options['bounds'][0][1:], options['bounds'][1][1:]]
                    auxiliar_options['bounds'] = [[reg_min]+rest[0]*(auxiliar_dimensions//3), [reg_max]+rest[1]*(auxiliar_dimensions//3)]
                else:
                    auxiliar_options['bounds'] = [options['bounds'][0]*(auxiliar_dimensions//3), options['bounds'][1]*(auxiliar_dimensions//3)]
                
            # random means
            #sol, best, bests_individuals, bests_fitness, worsts_individuals, worsts_fitness, means_fitness, std_fitness = test_elm(problem_type, data_to_train, hidden_layer_size, int(auxiliar_dimensions), initial_sigma, typea, auxiliar_options, elm, 0, check_conditions, gaussians_per_neuron, seed2 = seed_for_cmaes)
            
            # fixed means
            sol, _, bests_individuals, bests_fitness, worsts_individuals, worsts_fitness, means_fitness, std_fitness = test_elm(problem_type, data_to_train, hidden_layer_size, get_middle_points(auxiliar_options['bounds']), initial_sigma, typea, auxiliar_options, elm, 0, check_conditions, gaussians_per_neuron, seed2 = seed_for_cmaes)
            
            reg_value = 10**sol[0] if 'REG' in typea else None
            result_trained_function = set_function_and_execute_elm(elm, lambda x: gaussians(sol[1:], x), data_to_test, reg_value, problem_type)

            Logger.log("    Solution: " + str(sol))
            Logger.log("    Best: " + str(result_trained_function))
            Logger.log("    Bests_individuals: " + str(bests_individuals))
            Logger.log("    Bests_fitness: " + str(bests_fitness))
            Logger.log("    Worsts_individuals: " + str(worsts_individuals))
            Logger.log("    Worsts_fitness: " + str(worsts_fitness))
            Logger.log("    Means_fitness: " + str(means_fitness))
                
            printer.plot_one_object(bests_individuals, "Mejores individuos", "Iteraciones", "Genes de los individuos", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= 0, name_of_image= "best_individuals"), get_gen_labels(typea, dimensions), graphic_itf)
            printer.plot_one_object(bests_fitness, "Mejores valores de adaptacion", "Iteraciones", "Evolucion del mejor fitness", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= 0, name_of_image= "bests_fitness"), None, graphic_itf)
            printer.plot_one_object(worsts_individuals, "Peores individuos", "Iteraciones", "Genes de los individuos", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= 0, name_of_image= "worsts_individuals"), get_gen_labels(typea, dimensions), graphic_itf)
            printer.plot_one_object(worsts_fitness, "Peores valores de adaptacion", "Iteraciones", "Evolucion del peor fitness", path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= 0, name_of_image= "worsts_fitness"), None, graphic_itf)
            printer.plot_means_object(means_fitness, std_fitness, bests_fitness, "Media valores de adaptacion", "Iteraciones", "Evolucion de la media del fitness",path.format(domain = domain, sigma = initial_sigma, hidden_layer_size = hidden_layer_size, problem_type = typea, experiment_name = type_exe, number_dimensions = dimensions, test_number= 0, name_of_image= "means_fitness"), None, graphic_itf)
   

def main_test_cv(problem_type, type_exe, bounds_dict, pop_size, max_iter, sigmas, hidden_layer_sizes, number_of_gaussians, dataset, graphic_itf, check_conditions, gaus_per_neuron,scaler, path, **kwargs):
    matplotlib.use('Agg')
    typea = ComFunc.get_type(type_exe)
    counter=1
    bounds = get_bounds(typea, bounds_dict['bounds_reg'], bounds_dict['bounds_w'], bounds_dict['bounds_g'], bounds_dict['bounds_c'])
    for bound in bounds:
        options = {'bounds': bound, 'popsize_factor': pop_size, 'maxiter': max_iter}
        for sigma in sigmas:
            Logger.log("tests_with_sigma: " + str(sigma))
            for hidden_layer_size in hidden_layer_sizes:
                seed1 = np.random.RandomState(1234)
                try:
                    run_test_cv(problem_type, number_of_gaussians, hidden_layer_size, sigma, path=path, domain=counter, seed=seed1, scaler=scaler, options=options, dataset=dataset, type_exe=type_exe, typea=typea, graphic_itf=graphic_itf, check_conditions = check_conditions, gaussians_per_neuron = gaus_per_neuron, **kwargs)
                except Exception as e:
                    Logger.log(e)
                    Logger.log("No se ha podido completar el evolutivo")
        counter = counter+1