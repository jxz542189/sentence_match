import tensorflow as tf


def cross_entropy(logits, truth, mask=None):
    '''

    :param logits: [batch_size, passage_len]
    :param truth: [batch_size, passage_len]
    :param mask: [batch_size, passage_len]
    :return:
    '''
    if mask is not None:
        logits = tf.multiply(logits, mask)
    if mask is not None: logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev), -1)), -1))
    result = tf.multiply(truth, log_predictions)  # [batch_size, passage_len]
    if mask is not None: result = tf.multiply(result, mask)  # [batch_size, passage_len]
    return tf.sum(tf.multiply(-1.0, tf.reduce_sum(result, -1)))  # [batch_size]


