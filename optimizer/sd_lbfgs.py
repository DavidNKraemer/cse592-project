# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 02:07:39 2018

@author: vbigmouse
"""
<<<<<<< Updated upstream:sqn/optimizer/sd_lbfgs.py
from numpy import dot, zeros, asarray, sqrt
from math import isnan, isinf
=======
from numpy import dot, zeros, asarray, sqrt, power
>>>>>>> Stashed changes:optimizer/sd_lbfgs.py
from scipy.optimize import OptimizeResult
from time import time
from pprint import pprint

<<<<<<< Updated upstream:sqn/optimizer/sd_lbfgs.py
def harmonic_sequence(k, init_step_size):
    # For any nonnegative integer
    return init_step_size /(k+1)
#    return 10 / (k+1)
def sqrt_sequence(k, init_step_size):
    # For any nonnegative integer
    return init_step_size /(sqrt(k+1))
=======

def default_step_size(k):
    # For any nonnegative integer
    return power(k+1,-0.5)
>>>>>>> Stashed changes:optimizer/sd_lbfgs.py

class SdLBFGS():
    def __init__(self, func, initial_val, *args,
            max_iterations=1000,
            mem_size=10,
            batch_size=50,
<<<<<<< Updated upstream:sqn/optimizer/sd_lbfgs.py
            init_step_size=0.1,
            step_size=harmonic_sequence,
            delta=0.1,
            tol=1e-6,
=======
            step_size=default_step_size,
            delta=0.01,
            tol=1e-4,
>>>>>>> Stashed changes:optimizer/sd_lbfgs.py
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

<<<<<<< Updated upstream:sqn/optimizer/sd_lbfgs.py
        self._iteration_vals = [self._current_val]
        self._iteration_grads = [self._current_grad]
        self._iteration_objvals = [self._current_objval]

        self._start_time = time()
        self._iteration_runtimes = [0]
=======
        self._iteration_vals = []
        self._iteration_grad_errs = []
>>>>>>> Stashed changes:optimizer/sd_lbfgs.py

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
<<<<<<< Updated upstream:sqn/optimizer/sd_lbfgs.py
#        if self._iterations % 10 == 0:
#            print(f'norm of grad:={dot(self._current_grad.T, self._current_grad)}')

        if dot(self._current_grad.T, self._current_grad) <= self._tolerance:

=======
        self._iteration_grad_errs.append(dot(self._current_grad.T, self._current_grad))
        if self._iteration_grad_errs[-1]  <= self._tolerance:
>>>>>>> Stashed changes:optimizer/sd_lbfgs.py
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
<<<<<<< Updated upstream:sqn/optimizer/sd_lbfgs.py
        self._result['iteration_grads'] = self._iteration_grads
        self._result['iteration_objvals'] = self._iteration_objvals
        self._result['iteration_runtimes'] = self._iteration_runtimes
=======
        self._result['iteration_grad_errs'] = self._iteration_grad_errs
>>>>>>> Stashed changes:optimizer/sd_lbfgs.py

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
