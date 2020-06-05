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


with tf.Session(graph=tf.Graph()) as sess:
    sess.run(tf.global_variables_initializer())
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    model = Model(x, y)
    model.build_model()
    saver = tf.train.Saver(max_to_keep=1)

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

    feed_dict = {x: 10, y: 3}
    print(sess.run(model.op, feed_dict))

    with tf.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path+'savemodel')
    builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])
    builder.save()