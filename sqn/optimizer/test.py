#import tensorflow as tf
from sqn_optimizer_hooks import SQNOptimizer
from sd_lbfgs import SdLBFGS
from hw1_functions import weird_func
import numpy as np
from time import sleep
from pprint import pprint

def quadratic(x, order=1):
    if order == 0:
        return x * x
    if order==1:
        return x * x, 2 * x
#%%

optimizer = SdLBFGS(lambda x: weird_func(x, order=1), 1.,
        batch_size=2)

# for i in range(10):
#     optimizer.sqn_step()

result = optimizer.run()
import matplotlib.pyplot as plt

x = np.array(result['iteration_vals'])
vq = np.vectorize(weird_func)
y = vq(x, order=0)
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
