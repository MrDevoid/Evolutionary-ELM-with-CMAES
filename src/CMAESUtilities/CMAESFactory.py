from CMAESUtilities import CMAESMultipleGaus, CMAESRegMultipleGaus, CMAESNoReg, CMAESReg

def create_cmaes_problem(type_exe, hidden_layer_size, elm, data_to_test, seed2, problem_type, gaussians_per_neuron=1):
    if type_exe == "REG_MULTIPLE":
        return CMAESRegMultipleGaus.CMAESRegMultipleGaus("elm", hidden_layer_size, elm, data_to_test['X_train'], data_to_test['y_train'], data_to_test['X_test'], data_to_test['y_test'], seed = seed2, problem_type= problem_type, gaussians_per_neuron=gaussians_per_neuron)
    if type_exe == "MULTIPLE":
        return CMAESMultipleGaus.CMAESMultipleGaus("elm", hidden_layer_size, elm, data_to_test['X_train'], data_to_test['y_train'], data_to_test['X_test'], data_to_test['y_test'], seed = seed2, problem_type= problem_type, gaussians_per_neuron=gaussians_per_neuron)
    elif type_exe == "REG":
        return CMAESReg.CMAESReg("elm", hidden_layer_size, elm, data_to_test['X_train'], data_to_test['y_train'], data_to_test['X_test'], data_to_test['y_test'], seed = seed2, problem_type= problem_type)
    else:
        return CMAESNoReg.CMAESNoReg("elm", hidden_layer_size, elm, data_to_test['X_train'], data_to_test['y_train'], data_to_test['X_test'], data_to_test['y_test'], seed = seed2, problem_type= problem_type)
