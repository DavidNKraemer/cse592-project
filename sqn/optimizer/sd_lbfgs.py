# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 02:07:39 2018

@author: vbigmouse
"""
from numpy import dot, zeros, asarray, sqrt
from math import isnan, isinf
from scipy.optimize import OptimizeResult
from time import time
from pprint import pprint

def harmonic_sequence(k, init_step_size):
    # For any nonnegative integer
    return init_step_size /(k+1)
#    return 10 / (k+1)
def sqrt_sequence(k, init_step_size):
    # For any nonnegative integer
    return init_step_size /(sqrt(k+1))

class SdLBFGS():
    def __init__(self, func, initial_val, *args,
            max_iterations=1000,
            mem_size=10,
            batch_size=50,
            init_step_size=0.1,
            step_size=harmonic_sequence,
            delta=0.1,
            tol=1e-6,
            **kwargs):

        self._func = func
        self._initial_val = asarray(initial_val)
#        self._initial_val = self._initial_val.reshape(self._initial_val.shape[0])
        self._max_iterations = max_iterations
        self._max_mem_size = mem_size
        self._mem_size = 0
        self._batch_size = batch_size
        self._step_size = lambda k:step_size(k, self._init_step_size)
        self._init_step_size = init_step_size
        self._delta = delta
        self._tolerance = tol

        self._iterations = 0
        self._current_val = self._initial_val.copy()
        self._previous_val = zeros(self._initial_val.shape)
        self._current_grad = zeros(self._initial_val.shape)
        self._previous_grad = zeros(self._initial_val.shape)
        self._current_objval = asarray(self._func(self._current_val)[0])

        self._backward_errors = ShiftList(self._max_mem_size)
        self._ybars = ShiftList(self._max_mem_size)
        self._rhos = ShiftList(self._max_mem_size)

        self._iteration_vals = [self._current_val]
        self._iteration_grads = [self._current_grad]
        self._iteration_objvals = [self._current_objval]

        self._start_time = time()
        self._iteration_runtimes = [0]

        self._result = OptimizeResult()
        self._result['success'] = False



    def run(self):
        while not self._result['success'] and self._iterations < self._max_iterations:
#            pprint(self.__dict__, indent=2)
            self.sqn_step()
            self._iteration_runtimes.append(time() - self._start_time)


        return self.result()


    def sqn_step(self):
        # compute stochastic gradient

        value = zeros(self._current_objval.shape)
        grad = zeros(self._current_val.shape)
        for i in range(self._batch_size):
            v, g = self._func(self._current_val)
#            print(v)
            grad += g
            value += v

        grad /= self._batch_size
        value /= self._batch_size
#        grad = self._func(self._current_val)
        self._current_grad = grad
        self._current_objval = value



        # stopping criterion, sends back to self.run()
#        if self._iterations % 10 == 0:
#            print(f'norm of grad:={dot(self._current_grad.T, self._current_grad)}')

        if dot(self._current_grad.T, self._current_grad) <= self._tolerance:

            self._result['success'] = True

        # compute search direction
        direction = self.sdlbfgs_step()

        # update the iteration invariants
        self._previous_val = self._current_val.copy()
        self._current_val -= self._step_size(self._iterations) * direction
        self._iteration_vals.append(self._current_val.copy())
        self._iteration_grads.append(self._current_grad.copy())
        self._iteration_objvals.append(self._current_objval)
        self._previous_grad = self._current_grad.copy()
        self._iterations += 1


    def sdlbfgs_step(self):
        k = self._iterations
        p = self._mem_size

        s = self._current_val - self._previous_val
        y = self._current_grad - self._previous_grad
#        print(f's={s}, cur_val={self._current_val}')
#        print(f'y={y}, cur_grad={self._current_grad}')
        # compute theta
        sTy = dot(s.T, y)
        yTy = dot(y.T, y)
        sTs = dot(s.T, s)
        if sTy == 0 or isinf(yTy / sTy) or isnan(yTy / sTy):
            gamma = self._delta
        else:
            gamma = max(yTy / sTy, self._delta)


        if sTy < 0.25 * gamma * sTs:
            theta = (0.75 * gamma * sTs / (gamma * sTs - sTy)).item(0)
            rho = 1. / (0.25 * gamma * sTs)
        else:
            theta = 1.
            if sTy == 0:
                rho = 0
            else:
                rho = 1. / sTy

#        theta =1

        y_bar = theta * y + ((1. - theta) * gamma) * s

#        rho = 1. / dot(s.T, y_bar)



        if k == 0:
            return self._current_grad
        self.update_memory_vars(s, y_bar, rho)

        u = self._current_grad.copy()
        mus = []

        irange = range(min(p, k-1))
        for i in irange:
            mu = self._rhos[k-i-1] * dot(u.T, self._backward_errors[k-i-1])
            u -= mu * self._ybars[k-i-1]
            mus.append(mu)

        v = (1. / gamma) * u
        for i in irange:
            nu = self._rhos[k-p+i] * dot(v.T, self._ybars[k-p+i])
            v += (mus[p-i-1] - nu) * self._backward_errors[k-p+i]

        return v


    def update_memory_vars(self, backwards_error, forward_error, inverse_dp):
        self._backward_errors.append(backwards_error)
        self._ybars.append(forward_error)
        self._rhos.append(inverse_dp)


        self._mem_size = min(self._mem_size + 1, self._max_mem_size)
        if len(self._backward_errors) > self._max_mem_size:
            self._index_offset += 1


    def result(self):
        self._result['x'] = self._current_val.copy()
        self._result['status'] = 0
        self._result['message'] = 'SdLBFGS terminated with{} errors.'.format(
                'out' if self._result['success'] else ''
                )

        f, g = self._func(self._current_val)
        self._result['fun'] = f
        self._result['jac'] = g
        self._result['nfev'] = self._batch_size * self._iterations
        self._result['njev'] = self._batch_size * self._iterations
        self._result['nit'] = self._iterations
        self._result['iteration_vals'] = self._iteration_vals
        self._result['iteration_grads'] = self._iteration_grads
        self._result['iteration_objvals'] = self._iteration_objvals
        self._result['iteration_runtimes'] = self._iteration_runtimes

        return self._result



class ShiftList():
    def __init__(self, capacity):
        self._list = []
        self._capacity = capacity
        self._index_offset = 0

    def append(self, var):
        self._list.append(var)
        if len(self._list) > self._capacity:
            self._list.pop(0)
            self._index_offset += 1

    def __getitem__(self, key):
        return self._list[key - self._index_offset]

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return "[" + ",".join(f"{i + self._index_offset}::{item}" for i, item
                in enumerate(self._list)) + "]"

    def __repr__(self):
        return str(self)




# class SDLBFGS():
#     def __init__(self, func, datas, *args,
#             max_iteration=1000,
#             mem_size=10,
#             batch_size=50,
#             step_size=0.1,
#             delta=0.01,
#             **kwargs):
#         self.max_iteration = max_iteration
#         self.mem_size = mem_size
#         self.batch_size = batch_size
#         self.iteration = 0
#         self.step_size = step_size
#         self.delta = delta
#
#
#         self.func = func
#         self.initial_val = datas
#         self.num_sample = datas.shape[0]
#         self.num_feature = datas.shape[1] - 1
#         self.avg_func_value = None
#
#         self.st_grad = np.zeros(num_feature,1)
#         self.old_st_grad = np.zeros(num_feature,1)
#         self.weight = np.random.rand(num_feature,1)
#         self.old_weight = np.zeros(num_feature,1)
#
#         '''S is the update of weight'''
#         self.S = np.zeros((self.num_feature, mem_size))
#         '''Y is diff bewteen st_gradient'''
#         self.Y = np.zeros((self.num_feature, mem_size))
#
#         self.result = OptimizeResult()
#
#     def reset_S_Y(self):
#         self.S = np.zeros((self.num_feature, mem_size))
#         self.Y = np.zeros((self.num_feature, mem_size))
#
#     def save_S_Y(self, s, y):
# #        ind = self.iteration % self.mem_size
#         if self.iteration < self.mem_size:
#             self.S[:, self.iteration] = s
#             self.Y[:, self.iteration] = y
#         else:
#             self.S = np.concatenate(self.S, s, axis=1)
#             self.Y = np.concatenate(self.Y, y, axis=1)
#             self.S = numpy.delete(self.S,(0), axis=1)
#             self.Y = numpy.delete(self.Y,(0), axis=1)
#
#
#
#
#     def update_weight(self):
#
#         ''' calculate stochastic gradient g_{k-1} '''
#         sample_ind = rand.sample(range(1, self.func.num_sample), self.batch_size)
#         self.st_func_val, self.st_grad = self.Fun.get_st_subgradient(self.weight, sample_ind)
#
#         ''' save avg function value '''
#         self.avg_func_value = np.append(self.avg_func_value, st_func_val, axis=0)
#
#         ''' update weight '''
#         self.old_weight = self.weight
#         self.weight -= self.step_size * self.update_Hg(st_grad)
#         self.old_st_grad = self.st_grad
#
#         _, self.st_grad = self.Fun.get_st_subgradient(self.weight, sample_ind)
#
#         'update S, Y on same sample points'
#         s = self.weight - self.old_weight
#         y = self.st_grad - self.old_st_grad
#
#
#         ''' compute damping theta '''
#         sTy = s.T * y
#         yTy = y.T * y
#         sTs = s.T * s
#         gamma = max(yTy / sTy, self.delta)
#
#         if sTy < 0.25 * gamma * sTs:
#             theta = 0.75 * gamma * sTs / (gamma * sTs - sTy)
#         else:
#             theta = 1
#         y_bar = theta * y + (1 - theta) * gamma
#
#         self.save_S_Y(s, y_bar)
#
#     def update_Hg(self, st_grad):
#         ''' two loop'''
#         ''' first iteration Hg = g'''
#         if self.iteration == 0:
#             return st_grad
#
#         s = self.weight - self.old_weight
#         y = self.st_grad - self.old_st_grad
#
#
#         sTy = s.T * y
#         yTy = y.T * y
#         sTs = s.T * s
#
#
#
#         gamma = max(yTy / sTy, self.delta)
#         if sTy < 0.25 * gamma * sTs:
#             theta = 0.75 * gamma * sTs / (gamma * sTs - sTy)
#         else:
#             theta = 1
#         y_bar = theta * y + (1 - theta) * gamma
#
#         ' ro = 1 / sTy '
#         ' u = st_grad '
#         u = st_grad
#         ro = np.zeros(self.mem_size)
#         for i in range(min(self.mem_size, self.iteration) - 1, -1):
#             ro[i] = 1 / self.S[:i].T * self.Y[:,i]
#             mu = ro[i] * u.T * self.S[:,i]
#             u -= mu * self.Y[:,i]
#
#         v0 = 1 / gamma * u
#
#         for ind in range(0, min(self.mem_size, self.iteration)):
#             nu = ro[i] * v0.T * self.Y[:,i].T
#             v0 +=
#
#         return Hg
#
#     def package_result(self):
#         # ideally these updates happen organically in the algorithm updates.
#         # this is a TODO, but as of right now we can just port this back to the
#         # Optimizer
#         self.result['x'] = 1.
#         self.result['success'] = True
#         self.result['status'] = 0
#         self.result['message'] = 'SdLBFGS terminated with{} errors.'.format(
#                 'out' if self.result['success'] else ''
#                 )
#         self.result['fun'] = 1.
#         self.result['jac'] = 1.
#         self.result['nfev'] = 1
#         self.result['njev'] = 1
#         self.result['nit'] = 1
#
#         return result





