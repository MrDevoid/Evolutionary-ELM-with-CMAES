from tests_cmaes import *
from MnistScaler import MnistScaler
from os import path, makedirs
import utils.CommonFuncs as ComFunc
from sklearn.preprocessing import MinMaxScaler
import utils.ClassicFunctions as CF
import yaml
import sys

def read_config(path):
    with open(path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config[config['METHOD']]    

def get_scaler(scaler):
    if 'MNIST' in str.upper(scaler):
        return MnistScaler()
    else:
        return MinMaxScaler()

def gaussians(individual, val):
        sum = 0
        for i in range(0, len(individual)//3):
            sum += individual[i*3]*math.e**(-individual[i*3+1]*(val-individual[i*3+2])**2)
        return sum

def get_classic_function(classic):
    if 'relu' in str.lower(classic):
        return lambda x: CF.ClassicFunctions.relu(x)
    if 'sigmoid' in str.lower(classic):
        return lambda x: CF.ClassicFunctions.sigmoid(x)
    if 'swish' in str.lower(classic):
        return lambda x: CF.ClassicFunctions.swish(x)
    if 'tanh' in str.lower(classic):
        return lambda x: CF.ClassicFunctions.tanh(x)

def get_trained_function(individuals):
    return lambda x: gaussians(individuals, x)

def execute_method(config_file):
    if config_file['METHOD'] == 'main_test':
        bounds = {'bounds_reg':[tuple(config_file['BOUNDS_REG'])], 'bounds_w':[tuple(config_file['BOUNDS_W'])], 'bounds_g':[tuple(config_file['BOUNDS_G'])], 'bounds_c':[tuple(config_file['BOUNDS_C'])]}
        scaler = get_scaler(config_file['SCALER'])
        main_test(ComFunc.classification_or_regression(config_file['PROBLEM_TYPE']), config_file['TYPE_EXE'],
                  bounds,config_file['POP_SIZE'],config_file['MAX_ITER'],config_file['SIGMA'],config_file['HIDDEN_LAYER_SIZE'],
                  config_file['NUMBER_OF_GAUSSIANS'],config_file['DATASET'],config_file['GRAPHIC_ITF'],
                  config_file['CHECK_CONDITIONS'],config_file['GAUS_PER_NEURON'],scaler, config_file['PATH'], 
                  how_many=config_file.get('HOW_MANY'), test_size=config_file.get('TEST_SIZE'), string_y = config_file.get('STRING_Y'), 
                  header = config_file.get('HEADER'))
    elif config_file['METHOD'] == 'main_test_cv':
        bounds = {'bounds_reg':[tuple(config_file['BOUNDS_REG'])], 'bounds_w':[tuple(config_file['BOUNDS_W'])], 'bounds_g':[tuple(config_file['BOUNDS_G'])], 'bounds_c':[tuple(config_file['BOUNDS_C'])]}
        scaler = get_scaler(config_file['SCALER'])
        main_test_cv(ComFunc.classification_or_regression(config_file['PROBLEM_TYPE']), config_file['TYPE_EXE'],
                  bounds,config_file['POP_SIZE'],config_file['MAX_ITER'],config_file['SIGMA'],config_file['HIDDEN_LAYER_SIZE'],
                  config_file['NUMBER_OF_GAUSSIANS'],config_file['DATASET'],config_file['GRAPHIC_ITF'],
                  config_file['CHECK_CONDITIONS'],config_file['GAUS_PER_NEURON'],scaler, config_file['PATH'], 
                  how_many=config_file.get('HOW_MANY'), test_size=config_file.get('TEST_SIZE'), string_y = config_file.get('STRING_Y'), 
                  header = config_file.get('HEADER'))
    elif config_file['METHOD'] == 'print_several_activation_function_together':
        domain = range(config_file['RANGE_MIN'], config_file['RANGE_MAX'])
        print_several_activation_function_together(config_file['VALUES'], config_file['PATH'].format(problem_type = ComFunc.get_type(config_file['TYPE_EXE']), experiment_name=config_file['TYPE_EXE'], number_dimensions=config_file['NUMBER_OF_GAUSSIANS'][0]*3, image_name='image'), domain)
    elif config_file['METHOD'] == 'compare_trained_with_classics_activation_function':
        scaler = get_scaler(config_file['SCALER'])
        if 'REG' in ComFunc.get_type(config_file['TYPE_EXE']):
            compare_trained_with_classics_activation_function(ComFunc.classification_or_regression(config_file['PROBLEM_TYPE']),config_file['HIDDEN_LAYER'], config_file['DATASET'], list(map(lambda fun: get_trained_function(fun[1:]), config_file['TRAINED_FUNCTIONS'])), list(map(lambda fun: get_classic_function(fun),config_file['CLASSIC_FUNCTIONS'])), list(map(lambda ind: ind[0], config_file['TRAINED_FUNCTIONS'])), config_file['REG_FOR_CLASSIC'], scaler, string_y = config_file.get('STRING_Y'))
        else:
            compare_trained_with_classics_activation_function(ComFunc.classification_or_regression(config_file['PROBLEM_TYPE']),config_file['HIDDEN_LAYER'], config_file['DATASET'], list(map(lambda fun: get_trained_function(fun), config_file['TRAINED_FUNCTIONS'])), list(map(lambda fun: get_classic_function(fun),config_file['CLASSIC_FUNCTIONS'])), np.ones(len(config_file['TRAINED_FUNCTIONS'])), np.ones(len(config_file['CLASSIC_FUNCTIONS'])), scaler, string_y = config_file.get('STRING_Y'))
    elif config_file['METHOD'] == 'test_reg_fixed_functions':
        scaler = get_scaler(config_file['SCALER'])
        test_reg_fixed_functions(ComFunc.classification_or_regression(config_file['PROBLEM_TYPE']),config_file['HIDDEN_LAYER'], get_classic_function(config_file['CLASSIC_FUNCTION']), config_file['DATASET'], config_file['RANGE'],scaler)

CONFIG_PATH = 'config.yml'

def prepare_folder_and_stdout(config_file):
    if 'PATH' in config_file.keys():
        type_exe = ComFunc.get_type(config_file['TYPE_EXE'])
        for number_gaussian in np.multiply(3,config_file['NUMBER_OF_GAUSSIANS']):
            if 'HIDDEN_LAYER_SIZE' in config_file.keys():
                for hidden_layer in config_file['HIDDEN_LAYER_SIZE']:
                    path_to_check = '\\'.join(config_file['PATH'].split('\\')[:-1]).format(problem_type=type_exe,experiment_name=config_file['TYPE_EXE'], number_dimensions=number_gaussian, hidden_layer_size= hidden_layer)
                    if not path.exists(path_to_check):
                        makedirs(path_to_check)
            else:
                path_to_check = '\\'.join(config_file['PATH'].split('\\')[:-1]).format(problem_type=type_exe,experiment_name=config_file['TYPE_EXE'], number_dimensions=number_gaussian)
                if not path.exists(path_to_check):
                    makedirs(path_to_check)
        output_file = config_file['PATH'].split('{experiment_name')[0]+config_file['TYPE_EXE']+'\\'+config_file['TYPE_EXE']+'.txt'
        output_file = output_file.format(problem_type=type_exe)
    else:
        output_file = '.\\'+config_file['TYPE_EXE']+'.txt'
    sys.stdout = open(output_file, 'w')

if __name__ == '__main__':
    config_file = read_config(CONFIG_PATH)
    prepare_folder_and_stdout(config_file)
    execute_method(config_file)
    sys.stdout.close()