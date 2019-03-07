import tensorflow as tf
from network_funcs.mask_utils import exp_mask_for_high_rank, mask_for_high_rank


def pooling_with_mask(rep_tensor, rep_mask, method='max', scope=None):
    # rep_tensor have one more rank than rep_mask
    with tf.name_scope(scope or '%s_pooling' % method):

        if method == 'max':
            rep_tensor_masked = exp_mask_for_high_rank(rep_tensor, rep_mask)
            output = tf.reduce_max(rep_tensor_masked, -2)
        elif method == 'mean':
            rep_tensor_masked = mask_for_high_rank(rep_tensor, rep_mask)  # [...,sl,hn]
            rep_sum = tf.reduce_sum(rep_tensor_masked, -2)  #[..., hn]
            denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1, True)  # [..., 1]
            denominator = tf.where(tf.equal(denominator, tf.zeros_like(denominator, tf.int32)),
                                   tf.ones_like(denominator, tf.int32),
                                   denominator)
            output = rep_sum / tf.cast(denominator, tf.float32)
        else:
            raise AttributeError('No Pooling method name as %s' % method)
        return output
