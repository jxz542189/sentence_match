import tensorflow as tf
from network_funcs.nn_utils import bn_dense_layer
from network_funcs.dropout_utils import dropout
from network_funcs.mask_utils import exp_mask_for_high_rank, mask_for_high_rank
from network_funcs.nn_utils import linear
from network_funcs.logits_utils import get_logits
from network_funcs.softmax_utils import softsel


def disan(rep_tensor, rep_mask, scope=None,
          keep_prob=1., is_train=None, wd=0., activation='elu',
          tensor_dict=None, name=''):
    """

    :param rep_tensor:
    :param rep_mask:
    :param scope:
    :param keep_prob:
    :param is_train:
    :param wd:
    :param activation:
    :param tensor_dict:
    :param name:
    :return:
    """
    with tf.variable_scope(scope or 'DiSAN'):
        with tf.variable_scope('ct_attn'):
            fw_res = directional_attention_with_dense(
                rep_tensor, rep_mask, 'forward', 'dir_attn_fw',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_fw_attn')
            bw_res = directional_attention_with_dense(
                rep_tensor, rep_mask, 'backward', 'dir_attn_bw',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_bw_attn')

            seq_rep = tf.concat([fw_res, bw_res], -1)

        with tf.variable_scope('sent_enc_attn'):
            sent_rep = multi_dimensional_attention(
                seq_rep, rep_mask, 'multi_dimensional_attention',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_attn')
            return sent_rep


# --------------- supporting networks ----------------
def directional_attention_with_dense(rep_tensor, rep_mask, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def multi_dimensional_attention(rep_tensor, rep_mask, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


def traditional_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'traditional_attention'):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)

        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res


def expand_tile(units, axis):
    """Expand and tile tensor along given axis
    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2

    """
    assert axis in (1, 2)
    n_time_steps = tf.shape(units)[1]
    repetitions = [1, 1, 1, 1]
    repetitions[axis] = n_time_steps
    return tf.tile(tf.expand_dims(units, axis), repetitions)


def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """ Computes additive self attention for time series of vectors (with batch dimension)
        the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
        v is a learnable vector of n_hidden dimensionality,
        W_1 and W_2 are learnable [n_hidden, n_input_features] matrices

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
    """
    n_input_features = units.get_shape().as_list()[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    units_pairs = tf.concat([expand_tile(units, 1), expand_tile(units, 2)], 3)
    query = tf.layers.dense(units_pairs, n_hidden, activation=tf.tanh, kernel_initializer=INITIALIZER())
    attention = tf.nn.softmax(tf.layers.dense(query, 1), dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units, 1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation, kernel_initializer=INITIALIZER())
    return output


def multiplicative_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """ Computes multiplicative self attention for time series of vectors (with batch dimension)
        the formula: score(h_i, h_j) = <W_1 h_i,  W_2 h_j>,  W_1 and W_2 are learnable matrices
        with dimensionality [n_hidden, n_input_features], where <a, b> stands for a and b
        dot product

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
    """
    n_input_features = units.get_shape().as_list()[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    queries = tf.layers.dense(expand_tile(units, 1), n_hidden, kernel_initializer=INITIALIZER())
    keys = tf.layers.dense(expand_tile(units, 2), n_hidden, kernel_initializer=INITIALIZER())
    scores = tf.reduce_sum(queries * keys, axis=3, keep_dims=True)
    attention = tf.nn.softmax(scores, dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units, 1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation, kernel_initializer=INITIALIZER())
    return output


def self_attention_block(inputs, num_filters, seq_len, mask=None, num_heads=8,
                         scope="self_attention_ffn", reuse=None, is_training=True,
                         bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        l, L = sublayers
        # Self attention
        outputs = norm_fn(inputs, scope="layer_norm_1", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
            num_heads=num_heads, seq_len=seq_len, reuse=reuse,
            mask=mask, is_training=is_training, bias=bias, dropout=dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        # Feed-forward
        outputs = norm_fn(residual, scope="layer_norm_2", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name="FFN_1", reuse=reuse)
        outputs = conv(outputs, num_filters, True, None, name="FFN_2", reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l