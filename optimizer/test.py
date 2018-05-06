import tensorflow as tf
from sqn_optimizer_hooks import SQNOptimizer
from sd_lbfgs import SdLBFGS
from sd_lbfgs import harmonic_sequence as harmonic_seq
from sd_lbfgs import sqrt_sequence as sqrt_seq
from hw4_functions import svm_objective_function_stochastic as svm_func
from hw4_functions import svm_objective_function as svm_func_obj
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from pprint import pprint

def quadratic(x, order=1):
    if order == 0:
        return x * x
    if order==1:
        return x * x, 2 * x
def norm_z(x):
    x=(x-x.mean())/x.std()
#    m=np.median(x)
#    x.fillna(m,inplace=True)
    return x

def norm_m(x):
    x = (x - x.min() + 1) / (x.max() - x.min())
#    m = np.median(x)
#    x.fillna(m, inplace=True)
    return x

#%%
rawdata = np.loadtxt('HIGGS_subset.csv', delimiter=',')

labels = np.asmatrix(2 * rawdata[:, 0] -1).T
features = np.asmatrix(rawdata[:, 1:])
features = norm_z(features)

data = np.concatenate((features, labels), axis = 1)
d = features.shape[1]
w = np.zeros((d,1))
#%%
optimizer = SdLBFGS(lambda x: svm_func(x, order=1, data=data, minibatch_size=50), w,
        batch_size=50)

result = optimizer.run()
#%%
'''Logistic Resression test'''

'''General Setting'''
max_iterations=1000
points_to_plot=1000

from sd_lbfgs import SdLBFGS
from hw4_functions import cross_entropy_error as cross_enp_func

#%%
#labels = np.asmatrix(2 * rawdata[:, 0] -1).T
labels = np.asmatrix(rawdata[:, 0]).T
features = np.asmatrix(rawdata[:, 1:])
features = norm_z(features)

data = np.concatenate((features, labels), axis = 1)
d = features.shape[1]
w = np.zeros((d,1))


''''''
optimizer = SdLBFGS(lambda x: cross_enp_func(x, order=1, data=data, minibatch_size=1),
                    w,
                    batch_size=50,
                    delta=0.01,
                    init_step_size=0.001,
                    maxiiterations=10000)
result = optimizer.run()
sdlbfs_x =result['iteration_vals']
sdlbfs_values =result['iteration_objvals']
sdlbfs_runtimes =result['iteration_runtimes']

print('Solution found by sdLBFGS', sdlbfs_x[-1])
print('Objective function', cross_enp_func(sdlbfs_x[-1], order=0, data=data, minibatch_size=len(labels)))

#%%
'SGD'
from hw4_functions import cross_entropy_error as cross_enp_func
import algorithms as alg
obj_f = lambda x, order: cross_enp_func(x, order=order, data=data,  minibatch_size=50)
initial_x=np.zeros((d,1))

sgd_x, sgd_values, sgd_runtimes, sgd_xs = \
    alg.subgradient_descent(obj_f, initial_x, max_iterations, 0.1)

print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', obj_f(sgd_x,0))
sgd_its = len(sgd_runtimes)
sgd_values=[obj_f(sgd_xs[i],0) for i in range(0,sgd_its,int(sgd_its/min(sgd_its,points_to_plot)))]
#%%
'Ada'

ada_x, ada_values, ada_runtimes, ada_xs = alg.adagrad( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))

ada_its = len(ada_runtimes)
ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_its,int(ada_its/min(ada_its, points_to_plot)))]
#%%
'Subgradient descent'

sd_x, sd_values, sd_runtimes, sd_xs = alg.subgradient_descent( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by subgradient descent', sd_x)
print('Objective function', obj_f(sd_x,0))

sd_its = len(sd_runtimes)
sd_values=[obj_f(sd_xs[i],0) for i in range(0,sd_its,int(sd_its/min(sd_its, points_to_plot)))]
#Obj func vs time


#%%
'BFGS'
initial_x=np.zeros((d,1))
init_h =np.zeros((d,d))
obj_f = lambda x, order: cross_enp_func(x, order=order, data=data,  minibatch_size=1)
bfgs_x, bfgs_values, bfgs_runtimes, bfgs_xs = alg.bfgs(obj_f, initial_x, init_h, maximum_iterations=20)

print('Solution found by bfgs', bfgs_x)
print('Objective function', obj_f(bfgs_x,0))
bfgs_its = len(bfgs_runtimes)
bfgs_values = [obj_f(bfgs_xs[i],0) for i in range(0,bfgs_its,int(bfgs_its/min(bfgs_its, points_to_plot)))]

















#%%
'''Weird Function test'''

'''General Setting'''
max_iterations=1000
points_to_plot=1000
#%%
'SdLBFGS'
from sd_lbfgs import SdLBFGS
from hw1_functions import weird_func as weird_func

initial_x = 0.1
obj_f = lambda x: weird_func(x, order=1)
optimizer = SdLBFGS(obj_f, initial_x, batch_size=50, delta=0.1, max_iterations=max_iterations)
result = optimizer.run()

sdlbfs_x =result['iteration_vals']
sdlbfs_values =result['iteration_objvals']
sdlbfs_runtimes =result['iteration_runtimes']

print('Solution found by sdLBFGS', sdlbfs_x[-1])
print('Objective function', obj_f(sdlbfs_x[-1],0))

#%%
'SGD'
import algorithms as alg

obj_f = lambda x, order: weird_func(x, order)

sgd_x, sgd_values, sgd_runtimes, sgd_xs = \
    alg.subgradient_descent(obj_f, initial_x, max_iterations, 0.1)

print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', obj_f(sgd_x,0))
sgd_its = len(sgd_runtimes)
sgd_values=[obj_f(sgd_xs[i],0) for i in range(0,sgd_its,int(sgd_its/min(sgd_its,points_to_plot)))]
#%%
'Ada'

ada_x, ada_values, ada_runtimes, ada_xs = alg.adagrad( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))

ada_its = len(ada_runtimes)
ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_its,int(ada_its/min(ada_its, points_to_plot)))]
#%%
'Subgradient descent'

sd_x, sd_values, sd_runtimes, sd_xs = alg.subgradient_descent( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by subgradient descent', sd_x)
print('Objective function', obj_f(sd_x,0))

sd_its = len(sd_runtimes)
sd_values=[obj_f(sd_xs[i],0) for i in range(0,sd_its,int(sd_its/min(sd_its, points_to_plot)))]
#Obj func vs time


#%%
'BFGS'

init_h = 0.1
bfgs_x, bfgs_values, bfgs_runtimes, bfgs_xs = alg.bfgs(obj_f, initial_x, init_h, maximum_iterations=max_iterations)

print('Solution found by bfgs', bfgs_x)
print('Objective function', obj_f(bfgs_x,0))
bfgs_its = len(bfgs_runtimes)
bfgs_values = [obj_f(bfgs_xs[i],0) for i in range(0,bfgs_its,int(bfgs_its/min(bfgs_its, points_to_plot)))]


#%%
'''plot for weird function'''
plt.figure(figsize=(14,8))
#line_sdlbfgs, = plt.semilogx(np.array([i for i in range(len(sdlbfs_runtimes))]),
'''num of iterations to object function values'''
line_sdlbfgs, = plt.semilogx(np.array([i for i in range(len(sdlbfs_runtimes))]),
                         np.array(sdlbfs_values), linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')


#line_sgd, = plt.semilogx(np.array(sgd_runtimes[0::int(sgd_its/min(sgd_its, points_to_plot))]),
line_sgd, = plt.semilogx(np.array([i for i in range(len(sgd_runtimes))]),
                         np.array(sgd_values).flatten(), linewidth=2, color='r', dashes = [1, 1],
                         marker='o', label='SGD')
#line_sd, = plt.semilogx(sd_runtimes[0::int(sd_its/min(sd_its, points_to_plot))],
line_sd, = plt.semilogx(np.array([i for i in range(len(sd_runtimes))]),
                        np.array(sd_values).flatten(), linewidth=2, color='g',
                        marker='*', label='SD')

#line_ada, = plt.semilogx(ada_runtimes[0::int(sgd_its/min(ada_its, points_to_plot))],
line_ada, = plt.semilogx(np.array([i for i in range(len(ada_runtimes))]),
                        np.array(ada_values).flatten(), linewidth=2, color='b', dashes = [4, 1],
                        marker='x', label='AdaGrad')

#line_bfgs = plt.semilogx(bfgs_runtimes[0::int(bfgs_its/min(bfgs_its, points_to_plot))],
line_bfgs = plt.semilogx(np.array([i for i in range(len(bfgs_runtimes))]),
                        np.array(bfgs_values).flatten(), linewidth=2, color='g', dashes = [4, 1],
                        marker='x', label='BFGS')

plt.legend()
plt.show()

#%%
'''non-convex function'''

points_to_plot=1000
max_iterations = 10000

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
                    max_iterations=10000)

result = optimizer.run()

sdlbfs_x =result['iteration_vals']
sdlbfs_values =result['iteration_objvals']
sdlbfs_runtimes =result['iteration_runtimes']


print('Solution found by sdLBFGS', sdlbfs_x[-1])
print('Objective function', no_conv_func2(sdlbfs_x[-1],0))

sdlbfs_x =sdlbfs_x[0::int(len(sdlbfs_x)/min(len(sdlbfs_runtimes), points_to_plot))]
sdlbfs_values =sdlbfs_values[0::int(len(sdlbfs_values)/min(len(sdlbfs_runtimes), points_to_plot))]
sdlbfs_runtimes =sdlbfs_runtimes[0::int(len(sdlbfs_runtimes)/min(len(sdlbfs_runtimes), points_to_plot))]

'SGD'
import algorithms as alg

obj_f = lambda x, order: no_conv_func2(x, order)
initial_x = 2.
sgd_x, sgd_values, sgd_runtimes, sgd_xs = \
    alg.subgradient_descent(obj_f, initial_x, max_iterations, 0.01)

print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', obj_f(sgd_x,0))
sgd_its = len(sgd_runtimes)
sgd_values=[obj_f(sgd_xs[i],0) for i in range(0,sgd_its,int(sgd_its/min(sgd_its,points_to_plot)))]
#%%
'Ada'
obj_f = lambda x, order: no_conv_func2(x, order)
ada_x, ada_values, ada_runtimes, ada_xs = alg.adagrad( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))

ada_its = len(ada_runtimes)
ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_its,int(ada_its/min(ada_its, points_to_plot)))]
#%%
'BFGS'
obj_f = lambda x, order: no_conv_func2(x, order)
init_h = 0.1
bfgs_x, bfgs_values, bfgs_runtimes, bfgs_xs = alg.bfgs(obj_f, initial_x, init_h, maximum_iterations=max_iterations)

print('Solution found by bfgs', bfgs_x)
print('Objective function', obj_f(bfgs_x,0))
bfgs_its = len(bfgs_runtimes)
bfgs_values = [obj_f(bfgs_xs[i],0) for i in range(0,bfgs_its,int(bfgs_its/min(bfgs_its, points_to_plot)))]


#%%
plt.figure(figsize=(14,8))

'''num of iterations to object function values'''
sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
#sdlbfgs_X = np.array([i for i in range(len(sdlbfs_runtimes))])
sdlbfgs_Y = np.array(sdlbfs_values)
line_sdlbfgs, = plt.semilogx( sdlbfgs_X, sdlbfgs_Y, linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')

sgd_X = np.array([i for i in range(len(sgd_values))])
#sgd_X = np.array([i for i in range(0, len(sgd_runtimes),int(len(sgd_runtimes)/points_to_plot))])
sgd_Y = np.array(sgd_values).reshape(sgd_X.shape)
#line_sgd, = plt.semilogx(np.array(sgd_runtimes[0::int(sgd_its/min(sgd_its, points_to_plot))]),
line_sgd, = plt.semilogx(sgd_X,sgd_Y, linewidth=2, color='r', dashes = [1, 1],
                         marker='o', label='SGD')

#ada_X = np.array([i for i in range(0, len(ada_runtimes),int(len(ada_runtimes)/points_to_plot))])
ada_X = np.array([i for i in range(len(ada_values))])
ada_Y = np.array(ada_values).reshape(ada_X.shape)
#line_ada, = plt.semilogx(ada_runtimes[0::int(sgd_its/min(ada_its, points_to_plot))],
line_ada, = plt.semilogx(ada_X, ada_Y, linewidth=2, color='b', dashes = [4, 1],
                        marker='x', label='AdaGrad')

#bfgs_X = np.array([i for i in range(0, len(bfgs_runtimes),int(len(bfgs_runtimes)/points_to_plot))])
bfgs_X = np.array([i for i in range(len(bfgs_values))])
bfgs_Y = np.array(bfgs_values).reshape(bfgs_X.shape)
#line_bfgs = plt.semilogx(bfgs_runtimes[0::int(bfgs_its/min(bfgs_its, points_to_plot))],
line_bfgs = plt.semilogx(bfgs_X, bfgs_Y, linewidth=2, color='g', dashes = [4, 1],
                        marker='x', label='BFGS')

plt.title(r'non-convex function $x^{6}-2x^{5}+x^{3}-x^{2}+3x$')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()
#%%
plt.figure(figsize=(14,8))

'''num of iterations to object function values'''
sdlbfgs_X = np.array([i for i in range(0, len(sdlbfs_values))])
#sdlbfgs_X = np.array([i for i in range(len(sdlbfs_runtimes))])
sdlbfgs_Y = np.array(sdlbfs_values)
line_sdlbfgs, = plt.semilogx( sdlbfgs_X, sdlbfgs_Y, linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')

sgd_X = np.array([i for i in range(len(sgd_values))])
#sgd_X = np.array([i for i in range(0, len(sgd_runtimes),int(len(sgd_runtimes)/points_to_plot))])
sgd_Y = np.array(sgd_values).reshape(sgd_X.shape)
#line_sgd, = plt.semilogx(np.array(sgd_runtimes[0::int(sgd_its/min(sgd_its, points_to_plot))]),
line_sgd, = plt.semilogx(sgd_X,sgd_Y, linewidth=2, color='r', dashes = [1, 1],
                         marker='o', label='SGD')

#ada_X = np.array([i for i in range(0, len(ada_runtimes),int(len(ada_runtimes)/points_to_plot))])
ada_X = np.array([i for i in range(len(ada_values))])
ada_Y = np.array(ada_values).reshape(ada_X.shape)
#line_ada, = plt.semilogx(ada_runtimes[0::int(sgd_its/min(ada_its, points_to_plot))],
line_ada, = plt.semilogx(ada_X, ada_Y, linewidth=2, color='b', dashes = [4, 1],
                        marker='x', label='AdaGrad')

#bfgs_X = np.array([i for i in range(0, len(bfgs_runtimes),int(len(bfgs_runtimes)/points_to_plot))])
bfgs_X = np.array([i for i in range(len(bfgs_values))])
bfgs_Y = np.array(bfgs_values).reshape(bfgs_X.shape)
#line_bfgs = plt.semilogx(bfgs_runtimes[0::int(bfgs_its/min(bfgs_its, points_to_plot))],
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
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()



























#%%
'''tanh obj test'''

'''General Setting'''
max_iterations=1000
points_to_plot=1000

from sd_lbfgs import SdLBFGS
from hw4_functions import tanh_regul_error as tanh_obj

#%%
rawdata = np.loadtxt('HIGGS_subset.csv', delimiter=',')

labels = np.asmatrix(2 * rawdata[:, 0] -1).T
features = np.asmatrix(rawdata[:, 1:])
features = norm_z(features)

data = np.concatenate((features, labels), axis = 1)
d = features.shape[1]
w = np.zeros((d,1))



optimizer = SdLBFGS(lambda x: tanh_obj(x, order=1, data=data, minibatch_size=1),
                    initial_val=w,
                    batch_size=10,
                    delta=0.01,
                    step_size=harmonic_seq,
                    mem_size=20,
                    init_step_size=0.01,
                    max_iterations=1000)
result = optimizer.run()
sdlbfs_x =result['iteration_vals']
sdlbfs_values =result['iteration_objvals']
sdlbfs_runtimes =result['iteration_runtimes']
sdlbfs_grads = result['iteration_grads']

print('Solution found by sdLBFGS', sdlbfs_x[-1])
print('Objective function', tanh_obj(sdlbfs_x[-1], order=0, data=data, minibatch_size=len(labels)))


plt.figure(figsize=(14,8))
#line_sdlbfgs, = plt.semilogx(np.array([i for i in range(len(sdlbfs_runtimes))]),
'''num of iterations to object function values'''
pltX = np.array([i for i in range(len(sdlbfs_runtimes))])
#pltY = np.array(sdlbfs_values).reshape(pltX.shape)

pltY = np.linalg.norm(sdlbfs_grads, axis=1).reshape(pltX.shape)
line_sdlbfgs = plt.semilogx(pltX, pltY, linewidth=2, color='k', dashes = [1, 1],
                         marker='.', label='SdLBFGS')
plt.legend()
plt.grid()
plt.show()


#%%
'SGD'
from hw4_functions import cross_entropy_error as cross_enp_func
import algorithms as alg
obj_f = lambda x, order: cross_enp_func(x, order=order, data=data,  minibatch_size=50)
initial_x=np.zeros((d,1))

sgd_x, sgd_values, sgd_runtimes, sgd_xs = \
    alg.subgradient_descent(obj_f, initial_x, max_iterations, 0.1)

print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', obj_f(sgd_x,0))
sgd_its = len(sgd_runtimes)
sgd_values=[obj_f(sgd_xs[i],0) for i in range(0,sgd_its,int(sgd_its/min(sgd_its,points_to_plot)))]
#%%
'Ada'

ada_x, ada_values, ada_runtimes, ada_xs = alg.adagrad( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', obj_f(ada_x,0))

ada_its = len(ada_runtimes)
ada_values=[obj_f(ada_xs[i],0) for i in range(0, ada_its,int(ada_its/min(ada_its, points_to_plot)))]
#%%
'Subgradient descent'

sd_x, sd_values, sd_runtimes, sd_xs = alg.subgradient_descent( obj_f, initial_x, max_iterations, 0.1)
print('Solution found by subgradient descent', sd_x)
print('Objective function', obj_f(sd_x,0))

sd_its = len(sd_runtimes)
sd_values=[obj_f(sd_xs[i],0) for i in range(0,sd_its,int(sd_its/min(sd_its, points_to_plot)))]
#Obj func vs time


#%%
'BFGS'
initial_x=np.zeros((d,1))
init_h =np.zeros((d,d))
obj_f = lambda x, order: cross_enp_func(x, order=order, data=data,  minibatch_size=1)
bfgs_x, bfgs_values, bfgs_runtimes, bfgs_xs = alg.bfgs(obj_f, initial_x, init_h, maximum_iterations=20)

print('Solution found by bfgs', bfgs_x)
print('Objective function', obj_f(bfgs_x,0))
bfgs_its = len(bfgs_runtimes)
bfgs_values = [obj_f(bfgs_xs[i],0) for i in range(0,bfgs_its,int(bfgs_its/min(bfgs_its, points_to_plot)))]






#%%

x = np.array(result['iteration_vals'])
vq = np.vectorize(lambda x: svm_func_obj(x, order=0, data=data))

#optimizer = SdLBFGS(lambda x: weird_func(x, order=1), 1., batch_size=2)
#%%

#x = np.array(result['iteration_vals'])
#vq = np.vectorize(weird_func)
#y = vq(x, order=0)


vector = tf.Variable([7., 7.], 'vector')

 # Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector))
print(loss)
optimizer = SQNOptimizer(loss)

with tf.Session() as session:
    print("Inside the session.")
    session.run(tf.global_variables_initializer())
    optimizer.minimize(session)
     # The value of vector should now be [0., 0.].
