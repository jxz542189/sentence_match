import tensorflow as tf


def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    with tf.variable_scope(scope or "projection_layer"):
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs