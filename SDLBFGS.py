# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 02:07:39 2018

@author: vbigmouse
"""
import numpy as np
import random as rand
import Functions as Fun
from scipy.optimize import OptimizeResult

class SDLBFGS():
    def __init__(self, func, datas, *args, 
            max_iteration=1000, 
            mem_size=10, 
            batch_size=50, 
            step_size=0.1,
            delta=0.01, 
            **kwargs):
        self.max_iteration = max_iteration
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.iteration = 0
        self.step_size = step_size
        self.delta = delta


        self.Func = Fun(func, datas)
        self.num_sample = datas.shape[0]
        self.num_feature = datas.shape[1] - 1
        self.avg_func_value = None

        self.st_grad = np.zeros(num_feature,1)
        self.old_st_grad = np.zeros(num_feature,1)
        self.weight = np.random.rand(num_feature,1)
        self.old_weight = np.zeros(num_feature,1)


        '''S is the update of weight'''
        self.S = np.zeros((self.num_feature, mem_size))
        '''Y is diff bewteen st_gradient'''
        self.Y = np.zeros((self.num_feature, mem_size))

        self.result = OptimizeResult()

    def reset_S_Y(self):
        self.S = np.zeros((self.num_feature, mem_size))
        self.Y = np.zeros((self.num_feature, mem_size))

    def save_S_Y(self, s, y):
#        ind = self.iteration % self.mem_size
        if self.iteration < self.mem_size:
            self.S[:, self.iteration] = s
            self.Y[:, self.iteration] = y
        else:
            self.S = np.concatenate(self.S, s, axis=1)
            self.Y = np.concatenate(self.Y, y, axis=1)
            self.S = numpy.delete(self.S,(0), axis=1)
            self.Y = numpy.delete(self.Y,(0), axis=1)




    def update_weight(self):

        ''' calculate stochastic gradient g_{k-1} '''
        sample_ind = rand.sample(range(1, self.Func.num_sample), self.batch_size)
        self.st_func_val, self.st_grad = self.Fun.get_st_subgradient(self.weight, sample_ind)

        ''' save avg function value '''
        self.avg_func_value = np.append(self.avg_func_value, st_func_val, axis=0)

        ''' update weight '''
        self.old_weight = self.weight
        self.weight -= self.step_size * self.update_Hg(st_grad)
        self.old_st_grad = self.st_grad

        _, self.st_grad = self.Fun.get_st_subgradient(self.weight, sample_ind)

        'update S, Y on same sample points'
        s = self.weight - self.old_weight
        y = self.st_grad - self.old_st_grad


        ''' compute damping theta '''
        sTy = s.T * y
        yTy = y.T * y
        sTs = s.T * s
        gamma = max(yTy / sTy, self.delta)

        if sTy < 0.25 * gamma * sTs:
            theta = 0.75 * gamma * sTs / (gamma * sTs - sTy)
        else:
            theta = 1
        y_bar = theta * y + (1 - theta) * gamma

        self.save_S_Y(s, y_bar)

    def update_Hg(self, st_grad):
        ''' two loop'''
        ''' first iteration Hg = g'''
        if self.iteration == 0:
            return st_grad

        s = self.weight - self.old_weight
        y = self.st_grad - self.old_st_grad


        sTy = s.T * y
        yTy = y.T * y
        sTs = s.T * s



        gamma = max(yTy / sTy, self.delta)
        if sTy < 0.25 * gamma * sTs:
            theta = 0.75 * gamma * sTs / (gamma * sTs - sTy)
        else:
            theta = 1
        y_bar = theta * y + (1 - theta) * gamma

        ' ro = 1 / sTy '
        ' u = st_grad '
        u = st_grad
        ro = np.zeros(self.mem_size)
        for i in range(min(self.mem_size, self.iteration) - 1, -1):
            ro[i] = 1 / self.S[:i].T * self.Y[:,i]
            mu = ro[i] * u.T * self.S[:,i]
            u -= mu * self.Y[:,i]

        v0 = 1 / gamma * u

        for ind in range(0, min(self.mem_size, self.iteration)):
            nu = ro[i] * v0.T * self.Y[:,i].T
            v0 +=

        return Hg

    def package_result(self):
        self.result['x'] = 1.
        self.result['success'] = True
        self.result['status'] = 0
        self.result['message'] = 'SdLBFGS terminated with{} errors.'.format(
                'out' if self.result['success'] else ''
                )
        self.result['fun'] = 1.
        self.result['jac'] = 1.
        self.result['nfev'] = 1
        self.result['njev'] = 1
        self.result['nit'] = 1

        return result





