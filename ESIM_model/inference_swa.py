import tensorflow as tf
import collections
import re
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
import tempfile
from ESIM_model import Config
from ESIM_model.Utils import *
import logging
from datetime import datetime
import sys
from utils.data_util import *
from ESIM_model.esim_model_by_swa import ESIM
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_path = "log/log.{}".format(dt)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path)
logger = logging.getLogger(__name__)


jit_scope = tf.contrib.compiler.jit.experimental_jit_scope


with jit_scope():
    config = Config.ModelConfig()
    arg = config.arg

    arg.n_classes = 2
    print_log('CMD : python3 {0}'.format(' '.join(sys.argv)), file = logger)
    print_log('Training with following options :', file = logger)
    print_args(arg, logger)
    word_to_ids, id_to_vec = get_word_to_vec(file_name="vec_size100_mincount5.txt")
    arg.n_vocab = len(word_to_ids)

    model = ESIM(arg.seq_length, arg.n_vocab, arg.embedding_size, arg.hidden_size, arg.attention_size, arg.n_classes,
                 arg.batch_size, arg.optimizer, arg.l2, arg.clip_value)


    with tf.Session() as sess:
        with tf.gfile.GFile('tmp_dir/tmpz2cylzfu', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        output = tf.import_graph_def(graph_def,
                                     input_map={'premise:0': model.premise,
                                                'hypothesis:0': model.hypothesis,
                                                'premise_actual_length:0': model.premise_mask,
                                                'hypothesis_actual_length:0': model.hypothesis_mask,
                                                'dropout_keep_prob:0': model.dropout_keep_prob,
                                                'is_training:0': model.is_training,
                                                'use_moving_statistics:0': model.use_moving_statistics},
                                     return_elements=['composition/feed_forward/feed_foward_layer2/moving_free_batch_normalization/batchnorm/add_1:0'])
        print_log('... loading dataset ...', file=logger)
        premises_np = joblib.load('/root/PycharmProjects/sentence_match/data/premises_np_size100_mincount5.m')
        premises_mask = joblib.load('/root/PycharmProjects/sentence_match/data/premises_mask_size100_mincount5.m')
        hypothesis_np = joblib.load('/root/PycharmProjects/sentence_match/data/hypothesis_np_size100_mincount5.m')
        hypothesis_mask = joblib.load('/root/PycharmProjects/sentence_match/data/hypothesis_mask_size100_mincount5.m')
        labels_np = joblib.load('/root/PycharmProjects/sentence_match/data/labels_np_size100_mincount5.m')
        # labels_np = labels_np.astype(np.float32)
        i = 0
        premises_np_train, premises_np_test = train_test_split(premises_np, random_state=1234)
        premises_mask_train, premises_mask_test = train_test_split(premises_mask, random_state=1234)
        hypothesis_np_train, hypothesis_np_test = train_test_split(hypothesis_np, random_state=1234)
        hypothesis_mask_train, hypothesis_mask_test = train_test_split(hypothesis_mask, random_state=1234)
        labels_np_train, labels_np_test = train_test_split(labels_np, random_state=1234)
        arg.batch_size = 12
        batch_premises_np_test = premises_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_premises_mask_test = premises_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_np_test = hypothesis_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_mask_test = hypothesis_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
        result = sess.run(output, feed_dict={model.premise: batch_premises_np_test,
                             model.premise_mask: batch_premises_mask_test,
                             model.hypothesis: batch_hypothesis_np_test,
                             model.hypothesis_mask: batch_hypothesis_mask_test,
                             model.is_training: False,
                             model.use_moving_statistics: False,
                             model.dropout_keep_prob: 1.0})
        print(np.argmax(result))

