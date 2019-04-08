import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = ''


with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)

    op = tf.add(xy, b, name='op_to_store')

    sess.run(tf.global_variables_initializer())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                               ['op_to_store'])
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))

    with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())