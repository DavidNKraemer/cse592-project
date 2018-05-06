# -*- coding: utf-8 -*-
from sd_lbfgs import SdLBFGS
from sd_lbfgs import harmonic_sequence as harmonic_seq
from sd_lbfgs import sqrt_sequence as sqrt_seq
import matplotlib.pyplot as plt
import numpy as np



#%%
'''non-convex function'''

points_to_plot=100
max_iterations = 1000

from hw1_functions import no_conv_func2
obj_f = lambda x : no_conv_func2(x, order=1)


initial_x = 2.

optimizer = SdLBFGS(obj_f, initial_x,
                    batch_size=1,
                    delta=0.01,
                    step_size=sqrt_seq,
#                    step_size=harmonic_seq,
                    mem_size=20,
                    init_step_size=0.1,
                    max_iterations=1000)

result = optimizer.run()

sdlbfs_x =result['iteration_vals']
sdlbfs_values =result['iteration_objvals']
sdlbfs_runtimes =result['iteration_runtimes']
sdlbfs_grads = result['iteration_grads']

print('Solution found by sdLBFGS', sdlbfs_x[-1])
print('Objective function', no_conv_func2(sdlbfs_x[-1],0))

sdlbfs_its = len(sdlbfs_runtimes)
sdlbfs_x =sdlbfs_x[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]
sdlbfs_values =sdlbfs_values[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]
sdlbfs_runtimes =sdlbfs_runtimes[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]
sdlbfs_grads = sdlbfs_grads[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]

'SGD'
import algorithms as alg

obj_f = lambda x, order: no_conv_func2(x, order)
initial_x = 2.
sgd_x, sgd_values, sgd_runtimes, sgd_xs, sgd_grads = \
    alg.subgradient_descent(obj_f, initial_x, max_iterations, 0.01)

print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', obj_f(sgd_x,0))
sgd_its = len(sgd_runtimes)
sgd_values=[obj_f(sgd_xs[i],0) for i in range(0,sgd_its,int(sgd_its/min(sgd_its,points_to_plot)))]
sgd_xs = sgd_xs[0::int(sgd_its/min(sgd_its, points_to_plot))]
sgd_grads = sgd_grads[0::int(sgd_its/min(sgd_its, points_to_plot))]

'Ada'
obj_f = lambda x, order: no_conv_func2(x, order)
ada_x, ada_values, ada_runtimes, ada_xs, ada_grads = alg.adagrad( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))

ada_its = len(ada_runtimes)
ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_its,int(ada_its/min(ada_its, points_to_plot)))]
ada_xs = ada_xs[0::int(ada_its/min(ada_its, points_to_plot))]
ada_grads = ada_grads[0::int(ada_its/min(ada_its, points_to_plot))]

'BFGS'
obj_f = lambda x, order: no_conv_func2(x, order)
init_h = 0.1
bfgs_x, bfgs_values, bfgs_runtimes, bfgs_xs, bfgs_grads = alg.bfgs(obj_f, initial_x, init_h, maximum_iterations=max_iterations)

print('Solution found by bfgs', bfgs_x)
print('Objective function', obj_f(bfgs_x,0))
bfgs_its = len(bfgs_runtimes)

bfgs_values = [obj_f(bfgs_xs[i],0) for i in range(0,bfgs_its,int(bfgs_its/min(bfgs_its, points_to_plot)))]
bfgs_xs = bfgs_xs[0::int(bfgs_its/min(bfgs_its, points_to_plot))]
bfgs_grads = bfgs_grads[0::int(bfgs_its/min(bfgs_its, points_to_plot))]



#%%
'''num iterations to gradient norm'''
plt.figure(figsize=(14,8))


sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
sdlbfgs_Y = abs(np.array(sdlbfs_grads)).reshape(sdlbfgs_X.shape)
line_sdlbfgs, = plt.semilogx( sdlbfgs_X, sdlbfgs_Y, linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')

sgd_X = np.array([i for i in range(len(sgd_values))])
sgd_Y = abs(np.array(sgd_grads)).reshape(sgd_X.shape)
line_sgd, = plt.semilogx(sgd_X,sgd_Y, linewidth=2, color='r', dashes = [1, 1],
                         marker='o', label='SGD')

ada_X = np.array([i for i in range(len(ada_values))])
ada_Y = abs(np.array(ada_grads)).reshape(ada_X.shape)
line_ada, = plt.semilogx(ada_X, ada_Y, linewidth=2, color='b', dashes = [4, 1],
                        marker='x', label='AdaGrad')


bfgs_X = np.array([i for i in range(len(bfgs_values))])
bfgs_Y = abs(np.array(bfgs_grads)).reshape(bfgs_X.shape)
line_bfgs = plt.semilogx(bfgs_X, bfgs_Y, linewidth=2, color='g', dashes = [4, 1],
                        marker='x', label='BFGS')

plt.title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Norm of gradient')
plt.show()

#%%
'''num of iterations to object function values'''
plt.figure(figsize=(14,8))

sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
sdlbfgs_Y = np.array(sdlbfs_values)
line_sdlbfgs, = plt.semilogx( sdlbfgs_X, sdlbfgs_Y, linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')

sgd_X = np.array([i for i in range(len(sgd_values))])
sgd_Y = np.array(sgd_values).reshape(sgd_X.shape)

line_sgd, = plt.semilogx(sgd_X,sgd_Y, linewidth=2, color='r', dashes = [1, 1],
                         marker='o', label='SGD')


ada_X = np.array([i for i in range(len(ada_values))])
ada_Y = np.array(ada_values).reshape(ada_X.shape)

line_ada, = plt.semilogx(ada_X, ada_Y, linewidth=2, color='b', dashes = [4, 1],
                        marker='x', label='AdaGrad')


bfgs_X = np.array([i for i in range(len(bfgs_values))])
bfgs_Y = np.array(bfgs_values).reshape(bfgs_X.shape)

line_bfgs = plt.semilogx(bfgs_X, bfgs_Y, linewidth=2, color='g', dashes = [4, 1],
                        marker='x', label='BFGS')

plt.title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()
#%%
'''plot object function'''
plt.figure(figsize=(14,8))
pltX = np.array([i/10 for i in range(-15, 20, 1)])

pltY = no_conv_func2(pltX, 0)
plt.plot(pltX, pltY)
plt.title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()

