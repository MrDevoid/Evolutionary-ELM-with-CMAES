def get_type(type_exe):
    if 'REG' in str.upper(type_exe) and 'MULTIPLE' in str.upper(type_exe):
        return "REG_MULTIPLE"
    elif 'REG' in str.upper(type_exe):
        return "REG"
    elif 'MULTIPLE' in str.upper(type_exe):
        return "MULTIPLE"
    else:
        return "NORMAL"

def classification_or_regression(problem_type):
    if 'C' in str.upper(problem_type):
        return "CLASSIFICATION"
    else:
        return "REGRESSION"