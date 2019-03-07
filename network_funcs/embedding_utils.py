import tensorflow as tf
import numpy as np
from network_funcs.nn_utils import initializer as INITIALIZER

def embedding_layer(token_indices=None,
                    token_embedding_matrix=None,
                    n_tokens=None,
                    token_embedding_dim=None,
                    name: str = None,
                    trainable=True):
    """ Token embedding layer. Create matrix of for token embeddings.
        Can be initialized with given matrix (for example pre-trained
        with word2ve algorithm

    Args:
        token_indices: token indices tensor of type tf.int32
        token_embedding_matrix: matrix of embeddings with dimensionality
            [n_tokens, embeddings_dimension]
        n_tokens: total number of unique tokens
        token_embedding_dim: dimensionality of embeddings, typical 100..300
        name: embedding matrix name (variable name)
        trainable: whether to set the matrix trainable or not

    Returns:
        embedded_tokens: tf tensor of size [B, T, E], where B - batch size
            T - number of tokens, E - token_embedding_dim
    """
    if token_embedding_matrix is not None:
        tok_mat = token_embedding_matrix
        if trainable:
            Warning('Matrix of embeddings is passed to the embedding_layer, '
                    'possibly there is a pre-trained embedding matrix. '
                    'Embeddings paramenters are set to Trainable!')
    else:
        tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)
    tok_emb_mat = tf.Variable(tok_mat, name=name, trainable=trainable)
    embedded_tokens = tf.nn.embedding_lookup(tok_emb_mat, token_indices)
    return embedded_tokens


def character_embedding_network(char_placeholder: tf.Tensor,
                                n_characters: int =  None,
                                emb_mat: np.array = None,
                                char_embedding_dim: int = None,
                                filter_widths=(3, 4, 5, 7),
                                highway_on_top=False):
    """ Characters to vector. Every sequence of characters (token)
        is embedded to vector space with dimensionality char_embedding_dim
        Convolution plus max_pooling is used to obtain vector representations
        of words.

    Args:
        char_placeholder: placeholder of int32 type with dimensionality [B, T, C]
            B - batch size (can be None)
            T - Number of tokens (can be None)
            C - number of characters (can be None)
        n_characters: total number of unique characters
        emb_mat: if n_characters is not provided the emb_mat should be provided
            it is a numpy array with dimensions [V, E], where V - vocabulary size
            and E - embeddings dimension
        char_embedding_dim: dimensionality of characters embeddings
        filter_widths: array of width of kernel in convolutional embedding network
            used in parallel

    Returns:
        embeddings: tf.Tensor with dimensionality [B, T, F],
            where F is dimensionality of embeddings
    """
    if emb_mat is None:
        emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
    else:
        char_embedding_dim = emb_mat.shape[1]
    char_emb_var = tf.Variable(emb_mat, trainable=True)
    with tf.variable_scope('Char_Emb_Network'):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # Character embedding network
        conv_results_list = []
        for filter_width in filter_widths:
            conv_results_list.append(tf.layers.conv2d(c_emb,
                                                      char_embedding_dim,
                                                      (1, filter_width),
                                                      padding='same',
                                                      kernel_initializer=INITIALIZER))
        units = tf.concat(conv_results_list, axis=3)
        units = tf.reduce_max(units, axis=2)
        if highway_on_top:
            sigmoid_gate = tf.layers.dense(units,
                                           1,
                                           activation=tf.sigmoid,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            deeper_units = tf.layers.dense(units,
                                           tf.shape(units)[-1],
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            units = sigmoid_gate * units + (1 - sigmoid_gate) * deeper_units
            units = tf.nn.relu(units)
    return units

def _build_word_char_embeddings(batch_size, unroll_steps, projection_dim,
                                cnn_options, bidirectional):
    '''bilm-tf
    options contains key 'char_cnn': {

    'n_characters': 262,

    # includes the start / end characters
    'max_characters_per_token': 50,

    'filters': [
        [1, 32],
        [2, 32],
        [3, 64],
        [4, 128],
        [5, 256],
        [6, 512],
        [7, 512]
    ],
    'activation': 'tanh',

    # for the character embedding
    'embedding': {'dim': 16}

    # for highway layers
    # if omitted, then no highway layers
    'n_highway': 2,
    }
    '''

    filters = cnn_options['filters']
    n_filters = sum(f[1] for f in filters)
    max_chars = cnn_options['max_characters_per_token']
    char_embed_dim = cnn_options['embedding']['dim']
    n_chars = cnn_options['n_characters']
    if cnn_options['activation'] == 'tanh':
        activation = tf.nn.tanh
    elif cnn_options['activation'] == 'relu':
        activation = tf.nn.relu

    # the input character ids
    tokens_characters = tf.placeholder(tf.int32,
                                            shape=(batch_size, unroll_steps, max_chars),
                                            name='tokens_characters')
    # the character embeddings
    with tf.device("/cpu:0"):
        embedding_weights = tf.get_variable(
            "char_embed", [n_chars, char_embed_dim],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-1.0, 1.0)
        )
        # shape (batch_size, unroll_steps, max_chars, embed_dim)
        char_embedding = tf.nn.embedding_lookup(embedding_weights,
                                                     tokens_characters)

        if bidirectional:
            tokens_characters_reverse = tf.placeholder(tf.int32,
                                                            shape=(batch_size, unroll_steps, max_chars),
                                                            name='tokens_characters_reverse')
            char_embedding_reverse = tf.nn.embedding_lookup(
                embedding_weights, tokens_characters_reverse)

    # the convolutions
    def make_convolutions(inp, reuse):
        with tf.variable_scope('CNN', reuse=reuse) as scope:
            convolutions = []
            for i, (width, num) in enumerate(filters):
                if cnn_options['activation'] == 'relu':
                    # He initialization for ReLU activation
                    # with char embeddings init between -1 and 1
                    # w_init = tf.random_normal_initializer(
                    #    mean=0.0,
                    #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                    # )

                    # Kim et al 2015, +/- 0.05
                    w_init = tf.random_uniform_initializer(
                        minval=-0.05, maxval=0.05)
                elif cnn_options['activation'] == 'tanh':
                    # glorot init
                    w_init = tf.random_normal_initializer(
                        mean=0.0,
                        stddev=np.sqrt(1.0 / (width * char_embed_dim))
                    )
                w = tf.get_variable(
                    "W_cnn_%s" % i,
                    [1, width, char_embed_dim, num],
                    initializer=w_init,
                    dtype=tf.float32)
                b = tf.get_variable(
                    "b_cnn_%s" % i, [num], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

                conv = tf.nn.conv2d(
                    inp, w,
                    strides=[1, 1, 1, 1],
                    padding="VALID") + b
                # now max pool
                conv = tf.nn.max_pool(
                    conv, [1, 1, max_chars - width + 1, 1],
                    [1, 1, 1, 1], 'VALID')

                # activation
                conv = activation(conv)
                conv = tf.squeeze(conv, squeeze_dims=[2])

                convolutions.append(conv)

        return tf.concat(convolutions, 2)

    # for first model, this is False, for others it's True
    reuse = tf.get_variable_scope().reuse
    embedding = make_convolutions(char_embedding, reuse)

    token_embedding_layers = [embedding]

    if bidirectional:
        # re-use the CNN weights from forward pass
        embedding_reverse = make_convolutions(
            char_embedding_reverse, True)

    # for highway and projection layers:
    #   reshape from (batch_size, n_tokens, dim) to
    n_highway = cnn_options.get('n_highway')
    use_highway = n_highway is not None and n_highway > 0
    use_proj = n_filters != projection_dim

    if use_highway or use_proj:
        embedding = tf.reshape(embedding, [-1, n_filters])
        if bidirectional:
            embedding_reverse = tf.reshape(embedding_reverse,
                                           [-1, n_filters])

    # set up weights for projection
    if use_proj:
        assert n_filters > projection_dim
        with tf.variable_scope('CNN_proj') as scope:
            W_proj_cnn = tf.get_variable(
                "W_proj", [n_filters, projection_dim],
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                dtype=tf.float32)
            b_proj_cnn = tf.get_variable(
                "b_proj", [projection_dim],
                initializer=tf.constant_initializer(0.0),
                dtype=tf.float32)

    # apply highways layers
    def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
        carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
        transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
        return carry_gate * transform_gate + (1.0 - carry_gate) * x

    if use_highway:
        highway_dim = n_filters

        for i in range(n_highway):
            with tf.variable_scope('CNN_high_%s' % i) as scope:
                W_carry = tf.get_variable(
                    'W_carry', [highway_dim, highway_dim],
                    # glorit init
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                    dtype=tf.float32)
                b_carry = tf.get_variable(
                    'b_carry', [highway_dim],
                    initializer=tf.constant_initializer(-2.0),
                    dtype=tf.float32)
                W_transform = tf.get_variable(
                    'W_transform', [highway_dim, highway_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                    dtype=tf.float32)
                b_transform = tf.get_variable(
                    'b_transform', [highway_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32)

            embedding = high(embedding, W_carry, b_carry,
                             W_transform, b_transform)
            if bidirectional:
                embedding_reverse = high(embedding_reverse,
                                         W_carry, b_carry,
                                         W_transform, b_transform)
            token_embedding_layers.append(
                tf.reshape(embedding,
                           [batch_size, unroll_steps, highway_dim])
            )

    # finally project down to projection dim if needed
    if use_proj:
        embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
        if bidirectional:
            embedding_reverse = tf.matmul(embedding_reverse, W_proj_cnn) \
                                + b_proj_cnn
        token_embedding_layers.append(
            tf.reshape(embedding,
                       [batch_size, unroll_steps, projection_dim])
        )

    # reshape back to (batch_size, tokens, dim)
    if use_highway or use_proj:
        shp = [batch_size, unroll_steps, projection_dim]
        embedding = tf.reshape(embedding, shp)
        if bidirectional:
            embedding_reverse = tf.reshape(embedding_reverse, shp)

    # at last assign attributes for remainder of the model
    embedding = embedding
    if bidirectional:
        embedding_reverse = embedding_reverse
        return tf.concat([embedding, embedding_reverse], axis=-1)
    else:
        return embedding


def _build_word_embeddings(n_tokens_vocab, batch_size, unroll_steps,
                           projection_dim, bidirectional):

    # the input token_ids and word embeddings
    token_ids = tf.placeholder(tf.int32,
                           shape=(batch_size, unroll_steps),
                           name='token_ids')
    # the word embeddings
    with tf.device("/cpu:0"):
        embedding_weights = tf.get_variable(
            "embedding", [n_tokens_vocab, projection_dim],
            dtype=tf.float32,
        )
        embedding = tf.nn.embedding_lookup(embedding_weights,
                                            token_ids)

    # if a bidirectional LM then make placeholders for reverse
    # model and embeddings
    if bidirectional:
        token_ids_reverse = tf.placeholder(tf.int32,
                           shape=(batch_size, unroll_steps),
                           name='token_ids_reverse')
        with tf.device("/cpu:0"):
            embedding_reverse = tf.nn.embedding_lookup(
                embedding_weights, token_ids_reverse)