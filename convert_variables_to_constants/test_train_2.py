import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = ''


class Model(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def build_model(self):
        b = tf.Variable(1, name='b')
        xy = tf.multiply(self.x, self.y)
        self.op = tf.add(xy, b, name='op_to_store')

    def restore_model(self, sess, model_dir='ckpt/'):
        model_file = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, model_file)


with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    model = Model(x, y)
    model.build_model()
    saver = tf.train.Saver(max_to_keep=1)

    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'ckpt/model.ckpt')

