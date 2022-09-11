import matplotlib.pyplot as plt
import numpy as np
import math

def afunction(individual, val):
        sum = 0
        for i in range(0, len(individual)//3):
            sum += individual[i*3]*math.e**(-individual[i*3+1]*(val-individual[i*3+2])**2)
        return sum

def print_function_old(individuals, type_exe):
        eje_x = range(-10,10)
        eje_y = []
        for individual in individuals:
            eje_y += [list(map(lambda x: afunction(individual,x), eje_x))]
        fig,axs = plt.subplots(len(eje_y))
        for element in range(0, len(eje_y)):
            axs[element].plot(eje_x, element)
        fig.savefig("C:\\Users\\josem\\OneDrive - UNED\\TFM\\repo\\fig\\"+type_exe+"\\"+str(len(individuals[0]))+"-2.png")
        fig = plt.figure()
        for element in range(0, len(eje_y)):
            plt.plot(eje_x, eje_y[element])
        fig.savefig("C:\\Users\\josem\\OneDrive - UNED\\TFM\\repo\\fig\\"+type_exe+"\\"+str(len(individuals[0]))+"-3.png")


def print_function(individuals, fun, path, print_in_range = range(-10, 10)):
        eje_x = print_in_range
        eje_y = []
        for individual in individuals:
            eje_y += [list(map(lambda x: fun(individual,x), eje_x))]
        fig,axs = plt.subplots(len(eje_y))
        for element in range(0, len(eje_y)):
            axs[element].plot(eje_x, eje_y[element])
        fig.savefig(path)
        fig = plt.figure()
        for element in range(0, len(eje_y)):
            plt.plot(eje_x, eje_y[element])
        
        fig.savefig(path.split('.')[0] + '-2.png')

def plot_one_object(elements, title, xlabel, ylabel, path, labels = None, graphic_itf = True):
    fig = plt.figure()
    plt.plot(np.arange(len(elements)),elements, label=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not labels is None:
        plt.legend()
    if graphic_itf:
        plt.show()
    else:
        fig.savefig(path)
    plt.close()
    plt.clf()

def plot_means_object(elements, std, elements_to_compare, title, xlabel, ylabel, path, labels=None, graphic_itf = True):
    fig = plt.figure()
    plt.plot(np.arange(len(elements)), elements)
    plt.plot(np.arange(len(elements_to_compare)), elements_to_compare, color='red')
    plt.fill_between(np.arange(len(std)), elements-std, elements+std, facecolor = 'blue', alpha = 0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if graphic_itf:
        plt.show()
    else:
        fig.savefig(path)
    plt.close()
    plt.clf()