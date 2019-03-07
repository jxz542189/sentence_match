import tensorflow as tf
from network_funcs.activation_utils import selu
from network_funcs.dropout_utils import dropout
from network_funcs.dropout_utils import layer_dropout
from network_funcs.nn_utils import initializer, initializer_relu, regularizer
from network_funcs.batchnorm_utils import norm_fn


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1] #dc
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d] # (b,l,wl,d')
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d] # (b,l,d')
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            # (b*sn,sl,wl,dc)
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height)) #(b,l,d')
            outs.append(out)
        concat_out = tf.concat(outs,2) #(b,l,d)
        return concat_out


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len=None, scope="conv_block", is_training=True,
               reuse=None, bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.expand_dims(inputs, 2)
        l, L = sublayers
        for i in range(num_conv_layers):
            residual = outputs
            outputs = norm_fn(outputs, scope="layer_norm_%d" % i, reuse=reuse)
            if i % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
            outputs = depthwise_separable_convolution(outputs,
                kernel_size=(kernel_size, 1), num_filters=num_filters,
                scope="depthwise_conv_layers_%d" % i, is_training=is_training, reuse=reuse)
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
        return tf.squeeze(outputs, 2), l


def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype=tf.float32,
                        regularizer=regularizer,
                        initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope="depthwise_separable_convolution",
                                    bias=True, is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype=tf.float32,
                                        regularizer=regularizer,
                                        initializer=initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1, 1, shapes[-1], num_filters),
                                        dtype=tf.float32,
                                        regularizer=regularizer,
                                        initializer=initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides=(1, 1, 1, 1),
                                        padding="SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


INITIALIZER = tf.orthogonal_initializer


def stacked_cnn(units: tf.Tensor,
                n_hidden_list: list,
                filter_width=3,
                use_batch_norm=False,
                use_dilation=False,
                training_ph=None,
                add_l2_losses=False):
    """ Number of convolutional layers stacked on top of each other

    Args:
        units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
        n_hidden_list: list with number of hidden units at the ouput of each layer
        filter_width: width of the kernel in tokens
        use_batch_norm: whether to use batch normalization between layers
        use_dilation: use power of 2 dilation scheme [1, 2, 4, 8 .. ] for layers 1, 2, 3, 4 ...
        training_ph: boolean placeholder determining whether is training phase now or not.
            It is used only for batch normalization to determine whether to use
            current batch average (std) or memory stored average (std)
        add_l2_losses: whether to add l2 losses on network kernels to
                tf.GraphKeys.REGULARIZATION_LOSSES or not

    Returns:
        units: tensor at the output of the last convolutional layer
    """

    for n_layer, n_hidden in enumerate(n_hidden_list):
        if use_dilation:
            dilation_rate = 2 ** n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=INITIALIZER(),
                                 kernel_regularizer=tf.nn.l2_loss)
        if use_batch_norm:
            assert training_ph is not None
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)
    return units


def dense_convolutional_network(units: tf.Tensor,
                                n_hidden_list: list,
                                filter_width=3,
                                use_dilation=False,
                                use_batch_norm=False,
                                training_ph=None):
    """ Densely connected convolutional layers. Based on the paper:
        [Gao 17] https://arxiv.org/abs/1608.06993

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            filter_width: width of the kernel in tokens
            use_batch_norm: whether to use batch normalization between layers
            use_dilation: use power of 2 dilation scheme [1, 2, 4, 8 .. ] for layers 1, 2, 3, 4 ...
            training_ph: boolean placeholder determining whether is training phase now or not.
                It is used only for batch normalization to determine whether to use
                current batch average (std) or memory stored average (std)
        Returns:
            units: tensor at the output of the last convolutional layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
        """
    units_list = [units]
    for n_layer, n_filters in enumerate(n_hidden_list):
        total_units = tf.concat(units_list, axis=-1)
        if use_dilation:
            dilation_rate = 2 ** n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(total_units,
                                 n_filters,
                                 filter_width,
                                 dilation_rate=dilation_rate,
                                 padding='same',
                                 kernel_initializer=INITIALIZER())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)
        units_list.append(units)
    return units


def u_shape(units: tf.Tensor,
            n_hidden_list,
            filter_width=7,
            use_batch_norm=False,
            training_ph=None):
    """ Network architecture inspired by One Hundred layer Tiramisu.
        https://arxiv.org/abs/1611.09326. U-Net like.

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            filter_width: width of the kernel in tokens
            use_batch_norm: whether to use batch normalization between layers
            training_ph: boolean placeholder determining whether is training phase now or not.
                It is used only for batch normalization to determine whether to use
                current batch average (std) or memory stored average (std)
        Returns:
            units: tensor at the output of the last convolutional layer
                    with dimensionality [None, n_tokens, n_hidden_list[-1]]
    """
    units_for_skip_conn = []
    conv_net_params = {'filter_width': filter_width,
                       'use_batch_norm': use_batch_norm,
                       'training_ph': training_ph}

    # Go down the rabbit hole
    for n_hidden in n_hidden_list:
        units = stacked_cnn(units, [n_hidden], **conv_net_params)
        units_for_skip_conn.append(units)
        units = tf.layers.max_pooling1d(units, pool_size=2, strides=2, padding='same')

    units = stacked_cnn(units, [n_hidden], **conv_net_params)

    # Up to the sun light
    for down_step, n_hidden in enumerate(n_hidden_list[::-1]):
        units = tf.expand_dims(units, axis=2)
        units = tf.layers.conv2d_transpose(units, n_hidden, filter_width, strides=(2, 1), padding='same')
        units = tf.squeeze(units, axis=2)

        # Skip connection
        skip_units = units_for_skip_conn[-(down_step + 1)]
        if skip_units.get_shape().as_list()[-1] != n_hidden:
            skip_units = tf.layers.dense(skip_units, n_hidden)
        units = skip_units + units

        units = stacked_cnn(units, [n_hidden], **conv_net_params)
    return units


def stacked_highway_cnn(units: tf.Tensor,
                        n_hidden_list,
                        filter_width=3,
                        use_batch_norm=False,
                        use_dilation=False,
                        training_ph=None):
    """ Highway convolutional network. Skip connection with gating
        mechanism.

    Args:
        units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
        n_hidden_list: list with number of hidden units at the output of each layer
        filter_width: width of the kernel in tokens
        use_batch_norm: whether to use batch normalization between layers
        use_dilation: use power of 2 dilation scheme [1, 2, 4, 8 .. ] for layers 1, 2, 3, 4 ...
        training_ph: boolean placeholder determining whether is training phase now or not.
            It is used only for batch normalization to determine whether to use
            current batch average (std) or memory stored average (std)
    Returns:
        units: tensor at the output of the last convolutional layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
    """
    for n_layer, n_hidden in enumerate(n_hidden_list):
        input_units = units
        # Projection if needed
        if input_units.get_shape().as_list()[-1] != n_hidden:
            input_units = tf.layers.dense(input_units, n_hidden)
        if use_dilation:
            dilation_rate = 2 ** n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=INITIALIZER())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        sigmoid_gate = tf.layers.dense(input_units, 1, activation=tf.sigmoid, kernel_initializer=INITIALIZER())
        input_units = sigmoid_gate * input_units + (1 - sigmoid_gate) * units
        input_units = tf.nn.relu(input_units)
    units = input_units
    return units


def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype=tf.float32,
                        regularizer=regularizer,
                        initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope="depthwise_separable_convolution",
                                    bias=True, is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype=tf.float32,
                                        regularizer=regularizer,
                                        initializer=initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1, 1, shapes[-1], num_filters),
                                        dtype=tf.float32,
                                        regularizer=regularizer,
                                        initializer=initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides=(1, 1, 1, 1),
                                        padding="SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs



