# -*- coding: utf-8 -*-
from sys import path
path.append('../../optimizer/')

from sd_lbfgs import SdLBFGS

from sd_lbfgs import SdLBFGS
from sd_lbfgs import harmonic_sequence as harmonic_seq
from sd_lbfgs import sqrt_sequence as sqrt_seq

import algorithms as alg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context('paper')


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

sdlbfgs_x =result['iteration_vals']
sdlbfgs_values =result['iteration_objvals']
sdlbfgs_runtimes =result['iteration_runtimes']
sdlbfs_grads = result['iteration_grads']


print('Solution found by sdLBFGS', sdlbfgs_x[-1])
print('Objective function', no_conv_func2(sdlbfgs_x[-1],0))

sdlbfs_grads = sdlbfs_grads[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]


"""SGD"""


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
sgd_length = len(sgd_values)


"""Ada"""


obj_f = lambda x, order: no_conv_func2(x, order)
ada_x, ada_values, ada_runtimes, ada_xs, ada_grads = alg.adagrad( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))

ada_itr = len(ada_runtimes)

ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_itr,int(ada_itr/min(ada_itr, points_to_plot)))]
ada_xs = ada_xs[0::int(ada_itr/min(ada_itr, points_to_plot))]
ada_grads = ada_grads[0::int(ada_its/min(ada_its, points_to_plot))]

ada_length = len(ada_values)


"""BFGS"""


obj_f = lambda x, order: no_conv_func2(x, order)
init_h = 0.1
bfgs_x, bfgs_values, bfgs_runtimes, bfgs_xs, bfgs_grads = alg.bfgs(obj_f, initial_x, init_h, maximum_iterations=max_iterations)

print('Solution found by bfgs', bfgs_x)
print('Objective function', obj_f(bfgs_x,0))
bfgs_its = len(bfgs_runtimes)

bfgs_values = [obj_f(bfgs_xs[i],0) for i in range(0,bfgs_its,int(bfgs_its/min(bfgs_its, points_to_plot)))]
bfgs_xs = bfgs_xs[0::int(bfgs_its/min(bfgs_its, points_to_plot))]
bfgs_grads = bfgs_grads[0::int(bfgs_its/min(bfgs_its, points_to_plot))]

bfgs_length = len(bfgs_values)


#%%
'''num iterations to gradient norm'''
fig, axes = plt.subplots(1,3, figsize=(12,4))

plot_settings = {
        'linewidth' : 2,
        'dashes' : [1,1],
        }

sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
sdlbfgs_Y = abs(obj_f(np.array(sdlbfs_x).reshape(sdlbfgs_X.shape), 1)[1])
line_sdlbfgs, = plt.semilogx( sdlbfgs_X, sdlbfgs_Y, linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')

sdlbfgs_X = np.arange(sdlbfgs_length)
sdlbfgs_Y = abs(obj_f(np.array(sdlbfgs_x).reshape(sdlbfgs_X.shape), 1)[1])

line_sdlbfgs, = axes[0].semilogx(
        sdlbfgs_X, 
        sdlbfgs_Y, 
        marker='.',
        label='SdLBFGS', 
        **plot_settings)

sgd_X = np.arange(sgd_length)
sgd_Y = abs(obj_f(np.array(sgd_xs).reshape(sgd_X.shape),1)[1])

line_sgd, = axes[0].semilogx(
        sgd_X,
        sgd_Y, 
        marker='o', 
        label='SGD',
        **plot_settings)

ada_X = np.arange(ada_length)
ada_Y = abs(obj_f(np.array(ada_xs).reshape(ada_X.shape),1)[1])

line_ada, = axes[0].semilogx(
        ada_X, 
        ada_Y, 
        marker='x', 
        label='AdaGrad',
        **plot_settings)


bfgs_X = np.arange(bfgs_length)
bfgs_Y = abs(obj_f(np.array(bfgs_xs).reshape(bfgs_X.shape), 1)[1])

line_bfgs = axes[0].semilogx(
        bfgs_X, 
        bfgs_Y, 
        marker='x', 
        label='BFGS',
        **plot_settings)

#plt.title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
axes[0].legend()
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('$||\\nabla f(x) ||_2$')

#%%

'''num of iterations to object function values'''

sdlbfgs_Y = np.array(sdlbfgs_values)

line_sdlbfgs = axes[1].semilogx(
        sdlbfgs_X, 
        sdlbfgs_Y, 
        marker='v', 
        label='SdLBFGS',
        **plot_settings)

sgd_Y = np.array(sgd_values).reshape(sgd_X.shape)

line_sgd, = axes[1].semilogx(
        sgd_X,
        sgd_Y, 
        marker='o', 
        label='SGD',
        **plot_settings)


ada_Y = np.array(ada_values).reshape(ada_X.shape)

line_ada, = axes[1].semilogx(
        ada_X, 
        ada_Y, 
        marker='s', 
        label='AdaGrad',
        **plot_settings)


bfgs_Y = np.array(bfgs_values).reshape(bfgs_X.shape)

line_bfgs = axes[1].semilogx(
        bfgs_X, 
        bfgs_Y, 
        marker='D', 
        label='BFGS',
        **plot_settings)

#axes[1].title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
axes[1].legend()
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('$f(x)$')

#%%
'''plot object function'''

x_values = np.arange(-1.5,2.,.1)
y_values = no_conv_func2(x_values, 0)

axes[2].plot(x_values, y_values)
# axes[2].title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
axes[2].set_xlabel('$x$')
axes[2].set_ylabel('$f(x)$')

plt.suptitle('Non-convex function $f(x) = x^6 - 2x^5 + x^3 - x^2 + 3x$')
sns.despine(fig=fig)

fig.savefig('../../plots/nonconvex_results.eps', bbox_inches='tight')

