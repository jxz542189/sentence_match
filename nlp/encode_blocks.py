#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>


import tensorflow as tf

from nlp.nn import initializer, regularizer, spatial_dropout, get_lstm_init_state, dropout_res_layernorm


def LSTM_encode(seqs, causality=False, scope='lstm_encode_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        if causality:
            kwargs['direction'] = 'unidirectional'
        if 'num_units' not in kwargs or kwargs['num_units'] is None:
            kwargs['num_units'] = seqs.get_shape().as_list()[-1]
        batch_size = tf.shape(seqs)[0]
        _seqs = tf.transpose(seqs, [1, 0, 2])  # to T, B, D
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(**kwargs)
        init_state = get_lstm_init_state(batch_size, **kwargs)
        output = lstm(_seqs, init_state)[0]  # 2nd return is state, ignore
        return tf.transpose(output, [1, 0, 2])  # back to B, T, D


def TCN_encode(seqs, num_layers, scope='tcn_encode_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = [seqs]
        for i in range(num_layers):
            dilation_size = 2 ** i
            out = Res_DualCNN_encode(outputs[-1], dilation=dilation_size, scope='res_biconv_%d' % i, **kwargs)
            outputs.append(out)
        return outputs[-1]


def Res_DualCNN_encode(seqs, use_spatial_dropout=True, scope='res_biconv_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        out1 = CNN_encode(seqs, scope='first_conv1d', **kwargs)
        if use_spatial_dropout:
            out1 = spatial_dropout(out1)
        out2 = CNN_encode(out1, scope='second_conv1d', **kwargs)
        if use_spatial_dropout:
            out2 = CNN_encode(out2)
        return dropout_res_layernorm(seqs, out2, **kwargs)


def CNN_encode(seqs, filter_size=3, dilation=1,
               num_filters=None, direction='forward',
               causality=False,
               act_fn=tf.nn.relu,
               scope=None,
               reuse=None, **kwargs):
    input_dim = seqs.get_shape().as_list()[-1]
    num_filters = num_filters if num_filters else input_dim

    # add causality: shift the whole seq to the right
    if causality:
        direction = 'forward'
    padding = (filter_size - 1) * dilation
    if direction == 'forward':
        pad_seqs = tf.pad(seqs, [[0, 0], [padding, 0], [0, 0]])
        padding_scheme = 'VALID'
    elif direction == 'backward':
        pad_seqs = tf.pad(seqs, [[0, 0], [0, padding], [0, 0]])
        padding_scheme = 'VALID'
    elif direction == 'none':
        pad_seqs = seqs  # no padding, must set to SAME so that we have same length
        padding_scheme = 'SAME'
    else:
        raise NotImplementedError

    with tf.variable_scope(scope or 'causal_conv_%s_%s' % (filter_size, direction), reuse=reuse):
        return tf.layers.conv1d(
            pad_seqs,
            num_filters,
            filter_size,
            activation=act_fn,
            padding=padding_scheme,
            dilation_rate=dilation,
            kernel_initializer=initializer,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=regularizer)


def _reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return tf.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return tf.reverse(input_, axis=[seq_dim])


def elmo_encoder(source_embedding, sequence_length, num_layers, num_units):
    """
    from https://github.com/xueyouluo/fsauor2018
    :param source_embedding:
    :param sequence_length:
    :param num_layers:
    :param num_units:
    :return:
    """
    print("build elmo encoder")
    with tf.variable_scope("elmo_encoder") as scope:
        inputs = tf.transpose(source_embedding, [1, 0, 2])
        inputs_reverse = _reverse(
            inputs, seq_lengths=sequence_length,
            seq_dim=0, batch_dim=1)
        encoder_states = []
        outputs = [tf.concat([inputs, inputs], axis=-1)]
        fw_cell_inputs = inputs
        bw_cell_inputs = inputs_reverse
        for i in range(num_layers):
            with tf.variable_scope("fw_%d" % i) as s:
                cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units, use_peephole=False)
                fused_outputs_op, fused_state_op = cell(fw_cell_inputs, sequence_length=sequence_length,
                                                        dtype=inputs.dtype)
                encoder_states.append(fused_state_op)
            with tf.variable_scope("bw_%d" % i) as s:
                bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units, use_peephole=False)
                bw_fused_outputs_op_reverse, bw_fused_state_op = bw_cell(bw_cell_inputs,
                                                                         sequence_length=sequence_length,
                                                                         dtype=inputs.dtype)
                bw_fused_outputs_op = _reverse(
                    bw_fused_outputs_op_reverse, seq_lengths=sequence_length,
                    seq_dim=0, batch_dim=1)
                encoder_states.append(bw_fused_state_op)
            output = tf.concat([fused_outputs_op, bw_fused_outputs_op], axis=-1)
            if i > 0:
                fw_cell_inputs = output + fw_cell_inputs
                bw_cell_inputs = _reverse(
                    output, seq_lengths=sequence_length,
                    seq_dim=0, batch_dim=1) + bw_cell_inputs
            else:
                fw_cell_inputs = output
                bw_cell_inputs = _reverse(
                    output, seq_lengths=sequence_length,
                    seq_dim=0, batch_dim=1)
            outputs.append(output)

        final_output = None
        # embedding + num_layers
        n = 1 + num_layers
        scalars = tf.get_variable('scalar', initializer=tf.constant([1 / (n)] * n))
        weight = tf.get_variable('weight', initializer=tf.constant(0.001))

        soft_scalars = tf.nn.softmax(scalars)
        for i, output in enumerate(outputs):
            if final_output is None:
                final_output = soft_scalars[i] * tf.transpose(output, [1, 0, 2])
            else:
                final_output = final_output + soft_scalars[i] * tf.transpose(output, [1, 0, 2])

        final_outputs = weight * final_output
        final_state = tuple(encoder_states)
        return final_outputs, final_state

def gnmt_encoder(source_embedding, sequence_length, num_units, num_layers):
    print("build gnmt encoder")
    with tf.variable_scope("gnmt_encoder") as scope:
        inputs = tf.transpose(source_embedding, [1, 0, 2])
        inputs_reverse = _reverse(
            inputs, seq_lengths=sequence_length,
            seq_dim=0, batch_dim=1)
        encoder_states = []
        outputs = [inputs]

        with tf.variable_scope("fw") as s:
            cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units, use_peephole=False)
            fused_outputs_op, fused_state_op = cell(inputs, sequence_length=sequence_length,
                                                    dtype=inputs.dtype)
            encoder_states.append(fused_state_op)
            outputs.append(fused_outputs_op)

        with tf.variable_scope('bw') as s:
            bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units, use_peephole=False)
            bw_fused_outputs_op, bw_fused_state_op = bw_cell(inputs_reverse, sequence_length=sequence_length,
                                                             dtype=inputs.dtype)
            bw_fused_outputs_op = _reverse(
                bw_fused_outputs_op, seq_lengths=sequence_length,
                seq_dim=0, batch_dim=1)
            encoder_states.append(bw_fused_state_op)
            outputs.append(bw_fused_outputs_op)

        with tf.variable_scope("uni") as s:
            uni_inputs = tf.concat([fused_outputs_op, bw_fused_outputs_op], axis=-1)
            for i in range(num_layers - 1):
                with tf.variable_scope("layer_%d" % i) as scope:
                    uni_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units, use_peephole=False)
                    uni_fused_outputs_op, uni_fused_state_op = uni_cell(uni_inputs,
                                                                        sequence_length=sequence_length,
                                                                        dtype=inputs.dtype)
                    encoder_states.append(uni_fused_state_op)
                    outputs.append(uni_fused_outputs_op)
                    if i > 0:
                        uni_fused_outputs_op = uni_fused_outputs_op + uni_inputs
                    uni_inputs = uni_fused_outputs_op

        final_output = None
        # embedding + fw + bw + uni
        n = 3 + num_layers - 1
        scalars = tf.get_variable('scalar', initializer=tf.constant([1 / (n)] * n))
        weight = tf.get_variable('weight', initializer=tf.constant(0.001))

        soft_scalars = tf.nn.softmax(scalars)
        for i, output in enumerate(outputs):
            if final_output is None:
                final_output = soft_scalars[i] * tf.transpose(output, [1, 0, 2])
            else:
                final_output = final_output + soft_scalars[i] * tf.transpose(output, [1, 0, 2])

        final_outputs = weight * final_output
        final_state = tuple(encoder_states)
        return final_outputs, final_state
