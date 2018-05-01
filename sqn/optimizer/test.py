#import tensorflow as tf
from sqn_optimizer_hooks import SQNOptimizer
from sd_lbfgs import SdLBFGS
from hw1_functions import weird_func
from hw4_functions import svm_objective_function_stochastic as svm_func
import numpy as np
from time import sleep
from pprint import pprint

def quadratic(x, order=1):
    if order == 0:
        return x * x
    if order==1:
        return x * x, 2 * x

#%%
data = np.loadtxt('HIGGS_subset.csv', delimiter=',')

labels = np.asmatrix(2*data[:,0]-1).T
features = np.asmatrix(data[:,1:])
data = np.concatenate((features, labels), axis = 1)
d = features.shape[1]
w = np.zeros((d,1))
#%%

#optimizer = SdLBFGS(lambda x: weird_func(x, order=1), 1., batch_size=2)


optimizer = SdLBFGS(lambda x: svm_func(x, order=1, data=data, minibatch_size=50), w,
        batch_size=50)

# for i in range(10):
#     optimizer.sqn_step()

result = optimizer.run()
import matplotlib.pyplot as plt
#%%
from hw4_functions import svm_objective_function as svm_func_obj
x = np.array(result['iteration_vals'])
vq = np.vectorize(lambda x: svm_func_obj(x, order=0, data=data))

def pre_vectorized(x):
    return  svm_func_obj(x, order=0, data=data)

y = vq(pre_vectorized)
plt.plot(x,y, 'o')
plt.show()


# vector = tf.Variable([7., 7.], 'vector')
#
# # Make vector norm as small as possible.
# loss = tf.reduce_sum(tf.square(vector))
#
# print(loss)
#
# optimizer = SQNOptimizer(loss)
#
# with tf.Session() as session:
#     print("Inside the session.")
#     session.run(tf.global_variables_initializer())
#
#     optimizer.minimize(session)
#
#     # The value of vector should now be [0., 0.].
