######
######  This file includes different functions used in HW3
######

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random
import math



def svm_objective_function(w, data, order):
    labels = data[:,-1]
    features = data[:,:-1]

    n=len(labels)

    subgradient = np.zeros(w.shape)
#    result = []
    i = 0
#    sample_ind = random.sample(range(0, n), 10000)
    if order==0:
#        for w_v in w.T:
#            if i % 10 == 0:
        value = 0
        for i in range(0, n):
            value += int(max(1 - np.dot(labels[i], np.dot(features[i], w)), 0))
#        result.append(value)
        return value
#        return result
    elif order==1:
        for i in range(0, n):
            if labels[i].item(0) * features[i] * w < 1:
                subgradient += -np.multiply(labels[i], features[i].T) / n
            value += max(1 - labels[i] * np.dot(features[i].T, w), 0)
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

def svm_objective_function_stochastic(w, data, order, minibatch_size):

    labels = data[:,-1]
    features = data[:,:-1]
    n = len(labels)
    value = 0
    subgradient = np.zeros(w.shape)
    sample_ind = random.sample(range(0, n), minibatch_size)

    if order==0:
        for i in sample_ind:
            value += max(1 - labels[i] * features[i] * w , 0)
        return value
    elif order==1:
        for i in sample_ind:
            if labels[i].item(0) * features[i] * w < 1:
                subgradient += -np.multiply(labels[i], features[i].T) / minibatch_size
            value += max(1 - labels[i] * features[i] * w, 0)


        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

################################################################


def logrithm_loss(w, data, order, minibatch_size):
    labels = data[:,-1]
    features = data[:,:-1]
    n=len(labels)
    value = 0
    subgradient = np.zeros(w.shape)

    sample_ind = random.sample(range(0, n), minibatch_size)

    if order==0:
        for i in sample_ind:
            x = features[i].T
            value += math.log(1 + np.exp(- labels[i].item(0) * np.dot(w.T, x)))
        return value
    elif order==1:
        for i in sample_ind:
            x = features[i].T

            print(f'yWx: {yWx}')

            subgradient += (1 / (1 + math.exp(-yWx)) * (-labels[i].item(0) * x)) / minibatch_size
            value += math.log(1 + math.exp(-yWx)) / minibatch_size

        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

################################################################
def sigmoid(x):
#    print(-x)
    return 1/(1 + np.exp(-x))

def cross_entropy_error(w, data, order, minibatch_size):
    labels = data[:,-1]
    features = data[:,:-1]
    n=len(labels)

    value = 0
    subgradient = np.zeros(w.shape)

    sample_ind = random.sample(range(0, n), minibatch_size)

    if order==0:
        for i in sample_ind:
            x = features[i].T
            ywx = labels[i].item(0) * np.dot(w.T, x)
            if ywx > 0:
                value += np.log(1 + np.exp(-ywx))
            else:
                value += -ywx + np.log(1 + np.exp(ywx))
        return value
    elif order==1:
        for i in sample_ind:
            x = features[i].T
            y = labels[i].item(0)
            ywx = y * np.dot(w.T, x)
#            print(f'yWx: {ywx}')
            if ywx > 0:
                value += np.log(1 + np.exp(-ywx))
                subgradient += -labels[i].item(0)* x * sigmoid(ywx)
            else:
                value += -ywx + np.log(1 + np.exp(ywx))

                subgradient += -labels[i].item(0)* x + sigmoid(-ywx)

#            value += np.log(1 + np.exp(-yWx)) / minibatch_size

        return (value/ minibatch_size, subgradient/ minibatch_size)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

###################################################################
def tanh_regul_error(w, data, order, minibatch_size, regularizer=1e-4):
    labels = data[:,-1]
    features = data[:,:-1]
    n=len(labels)

    value = 0
    subgradient = np.zeros(w.shape)

    sample_ind = random.sample(range(0, n), minibatch_size)

    if order==0:
        for i in sample_ind:
            x = features[i].T
            ywx = labels[i].item(0) * np.dot(w.T, x)
            value += 1 - np.tanh(ywx)
        return value / minibatch_size + regularizer * np.dot(w.T, w)
    elif order==1:
        for i in sample_ind:
            x = features[i].T
            ywx = labels[i].item(0) * np.dot(w.T, x)
#            print(f'yWx: {ywx}')
            subgradient += labels[i].item(0) * (1 - np.power(np.tanh(ywx), 2).item(0)) * x
            value += 1 - np.tanh(ywx)
        subgradient /= minibatch_size
        value /= minibatch_size
        subgradient += 2 * regularizer * w
        value += regularizer * np.dot(w.T, w)
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")