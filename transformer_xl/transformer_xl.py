import tensorflow as tf
from transformer_xl import model


mem_len = 256#记忆维度
batch_size = 16#batch的大小
d_model = 500#模型维度
n_layer = 8 #transformer_xl层数


def transformer_xl(inp, mems,
                   n_token,
                   cutoffs=[],
                   n_layer=8,
                   d_model=500,
                   d_embed=500,
                   n_head=10,
                   d_head=50,
                   d_inner=1000,
                   dropout=0.1,
                   dropatt=0.1,
                   untie_r=False,
                   div_val=1,
                   tie_weight=True,
                   same_length=False,
                   clamp_len=-1,
                   proj_same_dim=True,
                   init="uniform",
                   init_range=0.1,
                   init_std=0.02,
                   proj_init_std=0.01,
                   proj_share_all_but_first=True,
                   is_training=True):
    '''
    reference: https://github.com/kimiyoung/transformer-xl/tree/master/tf
    :param inp: 输入数据,形状为(batch_size, sentence_len)
    :param mems: 记忆存在单元，形状为(mem_len, batch_size, d_model)
    :param n_token: 单词个数
    :param cutoffs: sentence切断边界，例如[0,100,200]将sentence切成两端
    :param n_layer: Number of layers.
    :param d_model: Dimension of the model.
    :param d_embed: Dimension of the embeddings.
    :param n_head: Number of attention heads.
    :param d_head: Dimension of each attention head.
    :param d_inner: Dimension of inner hidden size in positionwise feed-forward.
    :param dropout: Dropout rate.
    :param dropatt: Attention dropout rate.
    :param untie_r: untie r_w_bias and r_r_bias
    :param tie_weight:Tie embedding and softmax weight.
    :param div_val: Divide the embedding size by this val for each bin
    :param same_length: Same length attention
    :param clamp_len: Clamp length
    :param proj_same_dim: Project the bin with the same dimension.
    :param init: 当前支持uniform normal
    :param init_range: Initialization std when init is uniform.
    :param init_std: Initialization std when init is normal.
    :param proj_init_std: Initialization std for embedding projection.
    :param proj_share_all_but_first: True to share all but first projs, False not to share.
    :param is_training:
    :return:
    output:shape:
    '''
    inp = tf.transpose(inp, [1, 0])

    if init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-init_range,
          maxval=init_range,
          seed=None)
    elif init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    output, new_mems = model.transformer(
        dec_inp=inp,
        mems=mems,
        n_token=n_token,
        n_layer=n_layer,
        d_model=d_model,
        d_embed=d_embed,
        n_head=n_head,
        d_head=d_head,
        d_inner=d_inner,
        dropout=dropout,
        dropatt=dropatt,
        initializer=initializer,
        proj_initializer=proj_initializer,
        is_training=is_training,
        mem_len=mem_len,
        cutoffs=cutoffs,
        div_val=div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=same_length,
        clamp_len=clamp_len,
        use_tpu=False,
        untie_r=untie_r,
        proj_same_dim=proj_same_dim)
    output = tf.transpose(output, [1, 0])
    return output, new_mems

if __name__ == '__main__':
    mems_placeholders = [tf.placeholder(tf.float32,
                           [mem_len, batch_size, d_model])
            for _ in range(n_layer)]

    '''
    import numpy as np
    init_mems = [np.zeros([mem_len, batch_size, d_model], dtype=np.float32)
     for _ in range(n_layer)]
     最开始使用iniy_mems填充mems_placeholders
    output, new_mems = transformer_xl(inp, mems_placeholders, n_token)
    feed_dict = {mems_placeholders:new_mems}
    '''
