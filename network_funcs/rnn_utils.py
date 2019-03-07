import tensorflow as tf
from network_funcs.nn_utils import flatten, reconstruct
from network_funcs.nn_utils import initializer as INITIALIZER

from tensorflow.python.ops import nn_ops

INF = 1e30
INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)

def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bw_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                   dtype=None, parallel_iterations=None, swap_memory=False,
                   time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_inputs = tf.reverse(flat_inputs, [1]) if sequence_length is None \
        else tf.reverse_sequence(flat_inputs, sequence_length, 1)
    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)
    flat_outputs = tf.reverse(flat_outputs, [1]) if sequence_length is None \
        else tf.reverse_sequence(flat_outputs, sequence_length, 1)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state


class cudnn_gru:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def bi_rnn(units: tf.Tensor,
           n_hidden,
           cell_type='gru',
           seq_lengths=None,
           trainable_initial_states=False,
           use_peepholes=False,
           name='Bi-'):
    """ Bi directional recurrent neural network. GRU or LSTM

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden: list with number of hidden units at the ouput of each layer
            seq_lengths: length of sequences for different length sequences in batch
                can be None for maximum length as a length for every sample in the batch
            cell_type: 'lstm' or 'gru'
            trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
            use_peepholes: whether to use peephole connections (only 'lstm' case affected)
            name: what variable_scope to use for the network parameters

        Returns:
            units: tensor at the output of the last recurrent layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
            last_units: tensor of last hidden states for GRU and tuple
                of last hidden stated and last cell states for LSTM
                dimensionality of cell states and hidden states are
                similar and equal to [B x 2 * H], where B - batch
                size and H is number of hidden units
    """

    with tf.variable_scope(name + '_' + cell_type.upper()):
        if cell_type == 'gru':
            forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=INITIALIZER())
            backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=INITIALIZER())
            if trainable_initial_states:
                initial_state_fw = tf.tile(tf.get_variable('init_fw_h', [1, n_hidden]), (tf.shape(units)[0], 1))
                initial_state_bw = tf.tile(tf.get_variable('init_bw_h', [1, n_hidden]), (tf.shape(units)[0], 1))
            else:
                initial_state_fw = initial_state_bw = None
        elif cell_type == 'lstm':
            forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes, initializer=INITIALIZER())
            backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes, initializer=INITIALIZER())
            if trainable_initial_states:
                initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(
                    tf.tile(tf.get_variable('init_fw_c', [1, n_hidden]), (tf.shape(units)[0], 1)),
                    tf.tile(tf.get_variable('init_fw_h', [1, n_hidden]), (tf.shape(units)[0], 1)))
                initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(
                    tf.tile(tf.get_variable('init_bw_c', [1, n_hidden]), (tf.shape(units)[0], 1)),
                    tf.tile(tf.get_variable('init_bw_h', [1, n_hidden]), (tf.shape(units)[0], 1)))
            else:
                initial_state_fw = initial_state_bw = None
        else:
            raise RuntimeError('cell_type must be either "gru" or "lstm"s')
        (rnn_output_fw, rnn_output_bw), (fw, bw) = \
            tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                            backward_cell,
                                            units,
                                            dtype=tf.float32,
                                            sequence_length=seq_lengths,
                                            initial_state_fw=initial_state_fw,
                                            initial_state_bw=initial_state_bw)
    kernels = [var for var in forward_cell.trainable_variables +
               backward_cell.trainable_variables if 'kernel' in var.name]
    for kernel in kernels:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(kernel))
    return (rnn_output_fw, rnn_output_bw), (fw, bw)


def stacked_bi_rnn(units: tf.Tensor,
                   n_hidden_list,
                   cell_type='gru',
                   seq_lengths=None,
                   use_peepholes=False,
                   name='RNN_layer'):
    """ Stackted recurrent neural networks GRU or LSTM

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            seq_lengths: length of sequences for different length sequences in batch
                can be None for maximum length as a length for every sample in the batch
            cell_type: 'lstm' or 'gru'
            use_peepholes: whether to use peephole connections (only 'lstm' case affected)
            name: what variable_scope to use for the network parameters
        Returns:
            units: tensor at the output of the last recurrent layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
            last_units: tensor of last hidden states for GRU and tuple
                of last hidden stated and last cell states for LSTM
                dimensionality of cell states and hidden states are
                similar and equal to [B x 2 * H], where B - batch
                size and H is number of hidden units
    """
    for n, n_hidden in enumerate(n_hidden_list):
        with tf.variable_scope(name + '_' + str(n)):
            if cell_type == 'gru':
                forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
                backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
            elif cell_type == 'lstm':
                forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes)
                backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes)
            else:
                raise RuntimeError('cell_type must be either gru or lstm')

            (rnn_output_fw, rnn_output_bw), (fw, bw) = \
                tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                backward_cell,
                                                units,
                                                dtype=tf.float32,
                                                sequence_length=seq_lengths)
            units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
            if cell_type == 'gru':
                last_units = tf.concat([fw, bw], axis=1)
            else:
                (c_fw, h_fw), (c_bw, h_bw) = fw, bw
                c = tf.concat([c_fw, c_bw], axis=1)
                h = tf.concat([h_fw, h_bw], axis=1)
                last_units = (h, c)
    return units, last_units


def my_lstm_layer(input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False,
                  is_training=True, dropout_rate=0.2, use_cudnn=True):
    '''

    :param input_reps: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param input_lengths:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :param use_cudnn:
    :return:
    '''
    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        # if use_cudnn:
        #     inputs = tf.transpose(input_reps, [1, 0, 2])
        #     lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
        #                                           name="{}_cudnn_bi_lstm".format(scope_name),
        #                                           dropout=dropout_rate if is_training else 0)
        #     outputs, _ = lstm(inputs)
        #     outputs = tf.transpose(outputs, [1, 0, 2])
        #     f_rep = outputs[:, :, 0:lstm_dim]
        #     b_rep = outputs[:, :, lstm_dim: s * lstm_dim]
        # else:
        context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
        context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
        if is_training:
            context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw,
                                                                 output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw,
                                                                 output_keep_prob=(1 - dropout_rate))
        context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
        context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])
        (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(context_lstm_cell_fw, context_lstm_cell_bw, input_reps,
                                                            dtype=tf.float32, sequence_length=input_lengths)
        outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return f_rep, b_rep, outputs


def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr


