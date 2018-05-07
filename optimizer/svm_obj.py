# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from sd_lbfgs import SdLBFGS
from sd_lbfgs import harmonic_sequence as harmonic_seq
from sd_lbfgs import sqrt_sequence as sqrt_seq
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context('paper')

def norm_z(x):
    x=(x-x.mean())/x.std()
    return x

def correct_rate(w, features, labels):
    cor = 0
    for i in range(len(features)):
        if np.dot(w.T,features[i].T) * labels[i] >= 0:
            cor += 1
    return cor / len(features)

'''tanh obj test'''
from hw4_functions import svm_objective_function_stochastic as svm_obj

rawdata = np.loadtxt('HIGGS_subset.csv', delimiter=',')
labels = np.asmatrix(2 * rawdata[:, 0] -1).T
features = np.asmatrix(rawdata[:, 1:])
features = norm_z(features)

data = np.concatenate((features, labels), axis = 1)
d = features.shape[1]

#%%
'''General Setting'''
points_to_plot=100
max_iterations=10000


def const_step(k, initk):
    return initk

initial_x=np.zeros((d,1))

optimizer = SdLBFGS(lambda x: svm_obj(x, order=1, data=data, minibatch_size=1),
                    initial_val=initial_x,
                    batch_size=10,
                    tol=1e-6,
                    delta=0.1,
#                    step_size=harmonic_seq,
                    step_size=lambda k, initk:const_step(k, initk),
                    mem_size=20,
                    init_step_size=0.1,
                    max_iterations=1000)

result = optimizer.run()

sdlbfs_x =result['iteration_vals']
sdlbfs_values =result['iteration_objvals']
sdlbfs_runtimes =result['iteration_runtimes']
sdlbfs_grads = result['iteration_grads']


print('Solution found by sdLBFGS', sdlbfs_x[-1])
print('Objective function', svm_obj(sdlbfs_x[-1], order=0, data=data, minibatch_size=len(labels)))

sdlbfs_its = len(sdlbfs_runtimes)
sdlbfs_minind =  sdlbfs_values.index(min(sdlbfs_values))

sdlbfs_corr = correct_rate(sdlbfs_x[sdlbfs_minind], features, labels)
print(sdlbfs_corr)


sdlbfs_x =sdlbfs_x[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]
sdlbfs_values =sdlbfs_values[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]
sdlbfs_runtimes =sdlbfs_runtimes[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]
sdlbfs_grads = sdlbfs_grads[0::int(sdlbfs_its/min(sdlbfs_its, points_to_plot))]


#%%

import algorithms as alg
obj_f = lambda x, order: svm_obj(x, order=order, data=data, minibatch_size=10)
'SGD'
#initial_x = 2.
sgd_x, sgd_values, sgd_runtimes, sgd_xs, sgd_grads = \
    alg.subgradient_descent(obj_f, initial_x, max_iterations, 1)

print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', obj_f(sgd_x,0))
sgd_minind =  sgd_values.index(min(sgd_values))
sgd_corr = correct_rate(sgd_xs[sgd_minind], features, labels)
print(sgd_corr)
sgd_its = len(sgd_runtimes)
sgd_values=[obj_f(sgd_xs[i],0) for i in range(0,sgd_its,int(sgd_its/min(sgd_its,points_to_plot)))]
sgd_xs = sgd_xs[0::int(sgd_its/min(sgd_its, points_to_plot))]
sgd_grads = sgd_grads[0::int(sgd_its/min(sgd_its, points_to_plot))]

'Ada'
ada_x, ada_values, ada_runtimes, ada_xs, ada_grads = \
    alg.adagrad( obj_f, initial_x, max_iterations, 1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))
ada_minind =  ada_values.index(min(ada_values))
ada_corr = correct_rate(ada_xs[ada_minind], features, labels)
print(ada_corr)

ada_its = len(ada_runtimes)
ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_its,int(ada_its/min(ada_its, points_to_plot)))]
ada_xs = ada_xs[0::int(ada_its/min(ada_its, points_to_plot))]
ada_grads = ada_grads[0::int(ada_its/min(ada_its, points_to_plot))]
#%%
'LBFGS'

optimizer = SdLBFGS(lambda x: svm_obj(x, order=1, data=data, minibatch_size=1),
                    initial_val=initial_x,
                    batch_size=1,
                    tol=1e-7,
                    delta=1,
                    step_size=harmonic_seq,
                    mem_size=10,
                    init_step_size=0.1,
                    max_iterations=1000,
                    damp=False)

result = optimizer.run()

lbfgs_x =result['iteration_vals']
lbfgs_values =result['iteration_objvals']
lbfgs_runtimes =result['iteration_runtimes']
lbfgs_grads = result['iteration_grads']


print('Solution found by sdLBFGS', lbfgs_x[-1])
print('Objective function', svm_obj(sdlbfs_x[-1], order=0, data=data, minibatch_size=len(labels)))

lbfgs_its = len(lbfgs_runtimes)
lbfgs_minind =  lbfgs_values.index(min(lbfgs_values))

lbfgs_corr = correct_rate(lbfgs_x[lbfgs_minind], features, labels)
print(lbfgs_corr)


lbfgs_x =lbfgs_x[0::int(lbfgs_its/min(lbfgs_its, points_to_plot))]
lbfgs_values =lbfgs_values[0::int(lbfgs_its/min(lbfgs_its, points_to_plot))]
lbfgs_runtimes =lbfgs_runtimes[0::int(lbfgs_its/min(lbfgs_its, points_to_plot))]
lbfgs_grads = lbfgs_grads[0::int(lbfgs_its/min(lbfgs_its, points_to_plot))]
#%%
'''num iterations to gradient norm'''
fig, axes = plt.subplots(1,2, figsize=(12,4))

plot_settings = {
        'linewidth': 2,
        'dashes': [1,1]
        }


sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
sdlbfgs_Y = np.linalg.norm(np.array(sdlbfs_grads), axis=1).reshape(sdlbfgs_X.shape)
line_sdlbfgs, = axes[0].semilogx(
        sdlbfgs_X, 
        sdlbfgs_Y, 
        marker='v', 
        label='SdLBFGS',
        **plot_settings)

sgd_X = np.array([i for i in range(len(sgd_values))])
sgd_Y = np.linalg.norm(np.array(sgd_grads), axis=1).reshape(sgd_X.shape)
line_sgd, = axes[0].semilogx(
        sgd_X,
        sgd_Y, 
        marker='o', 
        label='SGD',
        **plot_settings)

ada_X = np.array([i for i in range(len(ada_values))])
ada_Y = np.linalg.norm(np.array(ada_grads),axis=1).reshape(ada_X.shape)
line_ada, = axes[0].semilogx(
        ada_X, 
        ada_Y, 
        marker='s', 
        label='AdaGrad',
        **plot_settings)


lbfgs_X = np.array([i for i in range(len(lbfgs_values))])
lbfgs_Y = np.linalg.norm(np.array(lbfgs_grads),axis=1).reshape(lbfgs_X.shape)
line_bfgs = axes[0].semilogx(
        lbfgs_X, 
        lbfgs_Y, 
        marker='D',
        label='LBFGS',
        **plot_settings)

#axes[0].set_title(r'SVM loss function $f(x) = \max(0, 1 - ywx)$')
axes[0].legend()
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('$|| \\nabla f(x) ||$')

#%%
'''num of iterations to object function values'''

sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
sdlbfgs_Y = np.array(sdlbfs_values).reshape(sdlbfgs_X.shape)
line_sdlbfgs, = axes[1].semilogx(
        sdlbfgs_X, 
        sdlbfgs_Y, 
        marker='v', 
        label='SdLBFGS',
        **plot_settings)

sgd_X = np.array([i for i in range(len(sgd_values))])
sgd_Y = np.array(sgd_values).reshape(sgd_X.shape)

line_sgd, = axes[1].semilogx(
        sgd_X,
        sgd_Y, 
        marker='o', 
        label='SGD',
        **plot_settings)


ada_X = np.array([i for i in range(len(ada_values))])
ada_Y = np.array(ada_values).reshape(ada_X.shape)

line_ada, = axes[1].semilogx(
        ada_X, 
        ada_Y, 
        marker='s', 
        label='AdaGrad',
        **plot_settings)


lbfgs_X = np.array([i for i in range(len(lbfgs_values))])
lbfgs_Y = np.array(lbfgs_values).reshape(lbfgs_X.shape)

line_bfgs = axes[1].semilogx(
        lbfgs_X,
        lbfgs_Y, 
        marker='D', 
        label='LBFGS',
        **plot_settings)

axes[1].legend()
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('$f(x)$')
plt.suptitle(r'SVM loss function $\max(0, 1 - ywx)$')
sns.despine(fig=fig)

fig.savefig('../plots/svm_hinge_loss.eps', bbox_inches='tight')

#%%


print(f'SdLBFGS correct rate: {sdlbfs_corr}')
print(f'SGD correct rate: {sgd_corr}')
print(f'ADAGfad correct rate: {ada_corr}')
print(f'LBFGS correct rate: {lbfgs_corr}')
