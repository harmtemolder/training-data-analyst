"""Halley's method for solving f(x) = 0

http://en.wikipedia.org/wiki/Halley%27s_method
http://code.activestate.com/recipes/577472-halleys-method-for-solving-equations/
https://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic

"""

import numpy as np
import tensorflow as tf
import sys

# tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


def f(x, a):
    """f(x) to solve. Assumes a 1D tensor of coefficients. E.g.
    [1.0, 2.0, 3.0] returns 1.0 + (2.0 * x) + (3.0 * (x ** 2))
    """
    result = 0
    
    for n in range(a.get_shape()[0]):
        result += a[n] * x ** n

    return result


def fp(x, a, k):
    """general numerical derivative functions
    """
    global h
    
    if k == 0:
        return f(x, a)
    else:
        return (fp(x + h, a, k - 1) - fp(x, a, k - 1)) / h


def xnew(x, a):
    """Calculate a new guess based on the previous value of x.
    """
    fx = f(x, a)
    fpx = fp(x, a, 1)
    
    xnew = (x - (2.0 * fx * fpx) / (2.0 * fpx * fpx - fx * fp(x, a, 2)), a)
    tf.Print(xnew, [x, a])
#     tf.print(xnew, output_stream=sys.stdout)
    
    return xnew
    
    
def condition(prev_x, a):
    """Returns True until the global variable eps is reached
    """
    global eps
    
    new_x, _ = xnew(prev_x, a)
    
    return abs(new_x - prev_x) > eps

    
global h
h = tf.constant(0.000001)  # Adjust h if xnew = nan

global eps
eps = tf.constant(0.1)  # tolerance

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # coefficients of f(x)
x = tf.constant(2.0)  # initial guess

result = tf.while_loop(
    cond=condition,
    body=xnew,
    loop_vars=(x, a))

with tf.Session() as sess:
    print(sess.run(result))

# n = 10000
# x = tf.constant(list(range(n)))
# condition = lambda xnew, x, eps: abs(xnew - x) > eps
# i, out = tf.while_loop(condition, b, (0, x))
# with tf.Session() as sess:
#     print(sess.run(i))  # prints [0] ... [9999]