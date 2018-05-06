# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:38:41 2018

@author: vbigmouse
"""

import numpy as np
import random as rand

class Functions():
    def __init__(self, func, datas):
        self.func = func
        self.datas = datas


    def get_st_value(self, weight, sample_ind):

        avg_value = None

        for ind in sample_ind:
            fvalue = self.func(weight, self.datas[ind:,-1], 0)
            avg_value += fvalue / len(sample_ind)

        return avg_value

    def get_st_subgradient(self, weight, sample_ind):

        avg_gradient = None
        avg_value = None

        for ind in sample_ind:
            fvalue, gradient = self.func(weight, self.datas[ind:,-1], 1)
            avg_gradient += gradient / len(sample_ind)
            avg_value += fvalue / len(sample_ind)

        return fvalue, avg_gradient

