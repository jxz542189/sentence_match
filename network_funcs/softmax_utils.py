import tensorflow as tf
from mask_utils import exp_mask


def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        out = tf.nn.softmax(logits,-1)
        return out


def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def softsel_with_dropout(target, logits, mask=None,
                         keep_prob=1., is_train=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "softsel_with_dropout"):
        a = softmax(logits, mask=mask)
        if keep_prob < 1.0:
            assert is_train is not None
            a = tf.cond(is_train, lambda: tf.nn.dropout(a, keep_prob), lambda: a)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out