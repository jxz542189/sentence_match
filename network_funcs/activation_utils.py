import tensorflow as tf
from tensorflow.python.framework import ops


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x, x_h = tf.split(x, 2, axis = -1)
    return tf.sigmoid(x) * x_h