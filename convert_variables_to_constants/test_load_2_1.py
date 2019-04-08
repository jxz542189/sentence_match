import tensorflow as tf



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with tf.gfile.GFile('tmp_dir/tmp5k63va3h', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    output = tf.import_graph_def(graph_def,
                                input_map={'x:0': x,
                                           'y:0': y},
                                 return_elements=['op_to_store:0'])
    # input_x = sess.graph.get_tensor_by_name("x:0")
    # input_y = sess.graph.get_tensor_by_name("y:0")
    # print(sess.run(tf.global_variables_initializer()))
    print(sess.run(output, feed_dict={x: 5, y: 5})[0])

