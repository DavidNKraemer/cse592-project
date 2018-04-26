
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow interface for third-party optimizers."""
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import numpy as np
 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import ExternalOptimizerInterface
import tensorflow as tf


class SQNOptimizer(ExternalOptimizerInterface):
  """Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.
 
  Example:
 
  ```python
  vector = tf.Variable([7., 7.], 'vector')
 
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
 
  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})
 
  with tf.Session() as session:
    optimizer.minimize(session)
 
  # The value of vector should now be [0., 0.].
  ```
 
  Example with simple bound constraints:
 
  ```python
  vector = tf.Variable([7., 7.], 'vector')
 
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
 
  optimizer = ScipyOptimizerInterface(
      loss, var_to_bounds={vector: ([1, 2], np.infty)})
 
  with tf.Session() as session:
    optimizer.minimize(session)
 
  # The value of vector should now be [1., 2.].
  ```
 
  Example with more complicated constraints:
 
  ```python
  vector = tf.Variable([7., 7.], 'vector')
 
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  # Ensure the vector's y component is = 1.
  equalities = [vector[1] - 1.]
  # Ensure the vector's x component is >= 1.
  inequalities = [vector[0] - 1.]
 
  # Our default SciPy optimization algorithm, L-BFGS-B, does not support
  # general constraints. Thus we use SLSQP instead.
  optimizer = ScipyOptimizerInterface(
      loss, equalities=equalities, inequalities=inequalities, method='SLSQP')
 
  with tf.Session() as session:
    optimizer.minimize(session)
 
  # The value of vector should now be [1., 1.].
  ```
  """
 
  _DEFAULT_METHOD = 'L-BFGS-B'
 
  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):
 
    def loss_grad_func_wrapper(x):
      # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
      loss, gradient = loss_grad_func(x)
      return loss, gradient.astype('float64')
 
    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)
 
    constraints = []
    for func, grad_func in zip(equality_funcs, equality_grad_funcs):
      constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
    for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
      constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})
 
    minimize_args = [loss_grad_func_wrapper, initial_val]
    minimize_kwargs = {
        'jac': True,
        'callback': step_callback,
        'method': method,
        'constraints': constraints,
        'bounds': packed_bounds,
    }
 
    for kwarg in minimize_kwargs:
      if kwarg in optimizer_kwargs:
        if kwarg == 'bounds':
          # Special handling for 'bounds' kwarg since ability to specify bounds
          # was added after this module was already publicly released.
          raise ValueError(
              'Bounds must be set using the var_to_bounds argument')
        raise ValueError(
            'Optimizer keyword arg \'{}\' is set '
            'automatically and cannot be injected manually'.format(kwarg))
 
    minimize_kwargs.update(optimizer_kwargs)
 
    result = SdLBFGS(*minimize_args, **minimize_kwargs)
 
    message_lines = [
        'Optimization terminated with:',
        '  Message: %s',
        '  Objective function value: %f',
    ]
    message_args = [result.message, result.fun]
    if hasattr(result, 'nit'):
      # Some optimization methods might not provide information such as nit and
      # nfev in the return. Logs only available information.
      message_lines.append('  Number of iterations: %d')
      message_args.append(result.nit)
    if hasattr(result, 'nfev'):
      message_lines.append('  Number of functions evaluations: %d')
      message_args.append(result.nfev)
    logging.info('\n'.join(message_lines), *message_args)
 
    return result['x']
 
 
def _accumulate(list_):
  total = 0
  yield total
  for x in list_:
    total += x
    yield total
 
 
def _get_shape_tuple(tensor):
  return tuple(dim.value for dim in tensor.get_shape())
 
 
def _prod(array):
  prod = 1
  for value in array:
    prod *= value
  return prod
 
 
def _compute_gradients(tensor, var_list):
  grads = gradients.gradients(tensor, var_list)
  # tf.gradients sometimes returns `None` when it should return 0.
  return [
      grad if grad is not None else array_ops.zeros_like(var)
      for var, grad in zip(var_list, grads)
  ]

def SdLBFGS():
    return 0.
