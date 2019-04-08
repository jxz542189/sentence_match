import tensorflow as tf


pb_file_path = ''
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'savemodel')
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
    print(ret)