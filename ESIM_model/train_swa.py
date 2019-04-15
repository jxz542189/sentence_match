import tensorflow as tf
import numpy as np
import os
from glob import glob
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse
import sys
from sklearn.externals import joblib
from utils.stochastic_weight_averaging import StochasticWeightAveraging
from ESIM_model.esim_model_by_swa import ESIM
from ESIM_model import Config
from ESIM_model.Utils import *
from utils.data_util import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import sys
import logging
from datetime import datetime

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_path = "log/log.{}".format(dt)
# log = open(arg.log_path, 'w')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path)
logger = logging.getLogger(__name__)


tf.reset_default_graph()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAIN_LOG_DIR = 'logs/'

PLOT_REPIOD = 20


COLORS = {
    'green': ['\033[32m', '\033[39m'],
    'red': ['\033[31m', '\033[39m']
}


def get_best_model(model_dir, model='best_model'):
    model_to_restore = None
    list_best_model_index = glob(os.path.join(model_dir, '{}.ckpt-*.index'.format(model)))
    if len(list_best_model_index) > 0:
        model_to_restore = list_best_model_index[0].split('.index')[0]
    return model_to_restore


def main():
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
    log_dir = os.path.join(MAIN_LOG_DIR, arg.log_path)
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    tf.set_random_seed(seed=42)

    print_log("... creating a TensorFlow session ...\n", file=logger)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print_log('... loading dataset ...', file=logger)
    premises_np = joblib.load('/root/PycharmProjects/sentence_match/data/premises_np_size100_mincount5.m')
    premises_mask = joblib.load('/root/PycharmProjects/sentence_match/data/premises_mask_size100_mincount5.m')
    hypothesis_np = joblib.load('/root/PycharmProjects/sentence_match/data/hypothesis_np_size100_mincount5.m')
    hypothesis_mask = joblib.load('/root/PycharmProjects/sentence_match/data/hypothesis_mask_size100_mincount5.m')
    labels_np = joblib.load('/root/PycharmProjects/sentence_match/data/labels_np_size100_mincount5.m')
    # labels_np = labels_np.astype(np.float32)
    premises_np_train, premises_np_test = train_test_split(premises_np, random_state=1234)
    premises_mask_train, premises_mask_test = train_test_split(premises_mask, random_state=1234)
    hypothesis_np_train, hypothesis_np_test = train_test_split(hypothesis_np, random_state=1234)
    hypothesis_mask_train, hypothesis_mask_test = train_test_split(hypothesis_mask, random_state=1234)
    labels_np_train, labels_np_test = train_test_split(labels_np, random_state=1234)
    model_vars = tf.trainable_variables()

    logits = model.logits

    update_bn_ops = tf.group(*tf.get_collection('UPDATE_BN_OPS'))
    reset_bn_ops = tf.group(*tf.get_collection('RESET_BN_OPS'))

    loss = model.loss
    acc = model.acc
    if arg.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(model.learning_rate)
    elif arg.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(model.learning_rate)
    elif arg.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(model.learning_rate)
    elif arg.optimizer == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(model.learning_rate)
    elif arg.optimizer == 'adagrad':
        opt = tf.train.AdagradOptimizer(model.learning_rate)
    elif arg.optimizer == "momentum":
        opt = tf.train.MomentumOptimizer(learning_rate=model.learning_rate, momentum=arg.momentum)
    elif arg.optimizer == "adamW":
        opt = tf.contrib.opt.AdamWOptimizer(weight_decay=arg.weight_decay, learning_rate=model.learning_rate,
                                            beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif arg.optimizer == "momentumW":
        opt = tf.contrib.opt.MomentumWOptimizer(weight_decay=arg.weight_decay, learning_rate=model.learning_rate,
                                                momentum=arg.momentum)
    else:
        ValueError('Unknown optimizer : {0}'.format(model.optimizer))

    if 'W' in model.optimizer:
        if arg.weight_decay_on == 'all':
            decay_var_list = tf.trainable_variables()
        elif arg.weight_decay_on == "kernels":
            decay_var_list = []
            for var in tf.trainable_variables():
                if 'kernel' in var.name:
                    decay_var_list.append(var)
        else:
            raise ValueError('Invalid --weight_decay_on : {}'.format(arg.weight_decay_on))
    global_step = tf.train.get_or_create_global_step()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables())
        if 'W' in arg.optimizer:
            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step,
                                           decay_var_list=decay_var_list, name="train_op")
        else:
            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')

    if arg.use_swa:
        with tf.name_scope('SWA'):
            swa = StochasticWeightAveraging()
            swa_op = swa.apply(var_list=model_vars)

            with tf.variable_scope('BackupVariables'):
                backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, initializer=var.initialized_value())
                               for var in model_vars]
            swa_to_weights = tf.group(*(tf.assign(var, swa.average(var).read_value()) for var in model_vars))
            save_weight_backups = tf.group(*(tf.assign(bck, var.read_value())
                                             for var, bck in zip(model_vars, backup_vars)))
            restore_weight_backups = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

    with tf.name_scope('METRICS'):
        acc_mean, acc_update_op = tf.metrics.mean(acc)
        loss_mean, loss_update_op = tf.metrics.mean(loss)

        acc_summary = tf.summary.scalar('TRAIN/acc', acc)
        loss_summary = tf.summary.scalar('TRAIN/loss', loss)

        acc_mean_summary = tf.summary.scalar('MEAN/acc', acc_mean)
        loss_mean_summary = tf.summary.scalar('MEAN/loss', loss_mean)

        lr_summary = tf.summary.scalar('lr', model.learning_rate)

        summaries_mean = tf.summary.merge([acc_mean_summary, loss_mean_summary], name='summaries_mean')
        summaries = [acc_summary, loss_summary, lr_summary]
        if arg.use_swa:
            n_models_summary = tf.summary.scalar('n_models', swa.n_models)
            summaries.append(n_models_summary)
        summaries = tf.summary.merge(summaries, name='summaries')

        with tf.name_scope('INIT_OPS'):
            global_init_op = tf.global_variables_initializer()
            local_init_op = tf.local_variables_initializer()

            sess.run(global_init_op)
            sess.run(local_init_op)

        with tf.name_scope('SAVERS'):
            best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            if arg.use_swa:
                best_saver_swa = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        with tf.name_scope('FILE_WRITERS'):
            writer_train = tf.summary.FileWriter(os.path.join(log_dir, 'train'), graph=sess.graph)
            writer_val = tf.summary.FileWriter(os.path.join(log_dir, 'val'))
            writer_val_bn = tf.summary.FileWriter(os.path.join(log_dir, 'val_bn'))
            writer_test = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

            if arg.use_swa:
                writer_val_swa = tf.summary.FileWriter(os.path.join(log_dir, 'val_swa'))
                writer_test_swa = tf.summary.FileWriter(os.path.join(log_dir, 'test_swa'))


        if arg.strategy_lr == "constant":
            def get_learning_rate(step, epoch, steps_per_epoch):
                return arg.init_lr
        elif arg.strategy_lr == "swa":
            def get_learning_rate(step, epoch, steps_per_epoch):
                if epoch < arg.epochs_before_swa:
                    return arg.init_lr

                if not arg.use_swa:
                    return arg.init_lr

                if step > int(0.9 * arg.num_epochs * steps_per_epoch):
                    return arg.alpha2_lr

                length_slope = int(0.9 * arg.num_epochs * steps_per_epoch) - arg.epochs_before_swa * steps_per_epoch
                return arg.alpha1_lr - ((arg.alpha1_lr - arg.alpha2_lr) / length_slope) * \
                       (step - arg.epochs_before_swa * steps_per_epoch)
        else:
            raise ValueError('Invalid --strategy_lr : {}'.format(arg.strategy_lr))

        ###################一定要注意，如果模型中没有使用batch norm，不要加入fit_bn_statistics
        def fit_bn_statistics(epoch, swa=False):
            sess.run(reset_bn_ops)


            if swa:
                desc = 'FIT STATISTICS @ EPOCH {} for SWA'.format(epoch)
            else:
                desc = 'FIT STATISTICS @ EPOCH {}'.format(epoch)
            sampleNums = len(premises_np_train)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums), desc=desc):
                batch_premises_np_train = premises_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_train = premises_mask_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_train = hypothesis_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_train = hypothesis_mask_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_train = labels_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]

                feed_dict = {model.premise: batch_premises_np_train,
                             model.premise_mask: batch_premises_mask_train,
                             model.hypothesis: batch_hypothesis_np_train,
                             model.hypothesis_mask: batch_hypothesis_mask_train,
                             model.learning_rate: arg.init_lr,
                             model.y: batch_labels_train,
                             model.is_training: True,
                             model.use_moving_statistics: True,
                             model.dropout_keep_prob: 1.0}
                sess.run([update_bn_ops], feed_dict=feed_dict)

        def inference(epoch, step, best_acc, best_step, best_epoch, with_moving_statistics=True):
            sess.run(local_init_op)
            sampleNums = len(premises_np_test)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums),
                      desc='VALIDATION (moving_statistics:{}) @ EPOCH {}'.format(with_moving_statistics, epoch)):
                batch_premises_np_test = premises_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_test = premises_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_test = hypothesis_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_test = hypothesis_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_test = labels_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]

                feed_dict = {model.premise: batch_premises_np_test,
                             model.premise_mask: batch_premises_mask_test,
                             model.hypothesis: batch_hypothesis_np_test,
                             model.hypothesis_mask: batch_hypothesis_mask_test,
                             model.y: batch_labels_test,
                             model.is_training: False,
                             model.use_moving_statistics: with_moving_statistics,
                             model.dropout_keep_prob: 1.0}
                sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)
            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])

            if with_moving_statistics:
                writer_val.add_summary(s, global_step=step)
                writer_val.flush()
            else:
                writer_val_bn.add_summary(s, global_step=step)
                writer_val_bn.flush()

            if acc_v > best_acc:
                color = COLORS['green']
                best_acc = acc_v
                best_step = step
                best_epoch = epoch
                ckpt_path = os.path.join(log_dir, 'best_model.ckpt')
                best_saver.save(sess, ckpt_path, global_step=step)
            else:
                color = COLORS['red']
            print_log("VALIDATION (moving_statistics:{}) @ EPOCH {} | without SWA : {}acc={:.4f}{}  loss={:.5f}".format(with_moving_statistics,
                                                                                                                    epoch,
                                                                                                                    color[0], acc_v, color[1],
                                                                                                                     loss_v), file=logger)

            return best_acc, best_step, best_epoch

        def inference_swa(epoch, step, best_acc_swa, best_step_swa, best_epoch_swa):
            sess.run(swa_op)
            sess.run(save_weight_backups)
            sess.run(swa_to_weights)

            fit_bn_statistics(epoch, swa=True)

            sess.run(local_init_op)

            sampleNums = len(premises_np_test)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums), desc='VALIDATION with SWA @ EPOCH {}'.format(epoch)):
                batch_premises_np_test = premises_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_test = premises_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_test = hypothesis_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_test = hypothesis_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_test = labels_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]

                feed_dict = {model.premise: batch_premises_np_test,
                             model.premise_mask: batch_premises_mask_test,
                             model.hypothesis: batch_hypothesis_np_test,
                             model.hypothesis_mask: batch_hypothesis_mask_test,
                             model.y: batch_labels_test,
                             model.is_training: False,
                             model.use_moving_statistics: False,
                             model.dropout_keep_prob: 1.0}
                sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
            writer_val_swa.add_summary(s, global_step=step)
            writer_val_swa.flush()

            if acc_v > best_acc_swa:
                color = COLORS['green']
                best_acc_swa = acc_v
                best_step_swa = step
                best_epoch_swa = epoch
                ckpt_path = os.path.join(log_dir, 'best_model_swa.ckpt')
                best_saver_swa.save(sess, ckpt_path, global_step=step)
            else:
                color = COLORS['red']

            print_log("VALIDATION @ EPOCH {} | with SWA : {}acc={:.4f}{}  loss={:.5f}".format(epoch,
                                                                                          color[0], acc_v, color[1],
                                                                                          loss_v), file=logger)
            sess.run(restore_weight_backups)
            return best_acc_swa, best_step_swa, best_epoch_swa


        best_acc = 0.
        best_step = 0
        best_epoch = 0

        best_acc_swa = 0.
        best_step_swa = 0
        best_epoch_swa = 0
        step = -1
        print_log("================================starting training model========================================", file=logger)
        best_acc, best_step, best_epoch = inference(0, 0, best_acc, best_step, best_epoch, with_moving_statistics=True)

        for epoch in range(1, arg.num_epochs+1):
            sess.run(local_init_op)
            sampleNums = len(premises_np_train)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums), desc='TRAIN @ EPOCH {}'.format(epoch)):
                step += 1
                batch_premises_np_train = premises_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_train = premises_mask_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_train = hypothesis_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_train = hypothesis_mask_train[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_train = labels_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]

                learning_rate = get_learning_rate(step, epoch, batchNums)
                feed_dict = {model.premise: batch_premises_np_train,
                             model.premise_mask: batch_premises_mask_train,
                             model.hypothesis: batch_hypothesis_np_train,
                             model.hypothesis_mask: batch_hypothesis_mask_train,
                             model.learning_rate: learning_rate,
                             model.y: batch_labels_train,
                             model.is_training: True,
                             model.use_moving_statistics: True,
                             model.dropout_keep_prob: arg.dropout_keep_prob}

                if step % PLOT_REPIOD == 0:
                    _, s, _, _ = sess.run([train_op, summaries, acc_update_op, loss_update_op], feed_dict=feed_dict)
                    writer_train.add_summary(s, global_step=step)
                else:
                    sess.run([train_op, acc_update_op, loss_update_op], feed_dict=feed_dict)

            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
            writer_train.add_summary(s, global_step=step)
            writer_train.flush()
            print_log("TRAIN @ EPOCH {} | : acc={:.4f}  loss={:.5f}".format(epoch, acc_v, loss_v), file=logger)

            best_acc, best_step, best_epoch = inference(epoch, step, best_acc, best_step, best_epoch,
                                                        with_moving_statistics=True)
            fit_bn_statistics(epoch, swa=False)
            best_acc, best_step, best_epoch = inference(epoch, step, best_acc, best_step, best_epoch,
                                                        with_moving_statistics=False)

            if epoch >= arg.epochs_before_swa \
                and arg.use_swa \
                and (epoch -arg.epochs_before_swa) % arg.cycle_length == 0:
                best_acc_swa, best_step_swa, best_epoch_swa = inference_swa(epoch, step, best_acc_swa,
                                                                            best_step_swa, best_epoch_swa)

        if best_acc > 0:
            print_log("Load best model without SWA | ACC={:.5f} from epoch={}".format(best_acc, best_epoch), file=logger)
            model_to_restore = get_best_model(log_dir, model='best_model')
            if model_to_restore is not None:
                best_saver.restore(sess, model_to_restore)
            else:
                print_log("Impossible to load best model ...", file=logger)

            sess.run(local_init_op)
            sampleNums = len(premises_np_test)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums), desc='TEST'):
                batch_premises_np_test = premises_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_test = premises_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_test = hypothesis_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_test = hypothesis_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_test = labels_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]

                feed_dict = {model.premise: batch_premises_np_test,
                             model.premise_mask: batch_premises_mask_test,
                             model.hypothesis: batch_hypothesis_np_test,
                             model.hypothesis_mask: batch_hypothesis_mask_test,
                             model.y: batch_labels_test,
                             model.is_training: False,
                             model.use_moving_statistics: False,
                             model.dropout_keep_prob: 1.0}
                sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)
            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
            writer_test.add_summary(s, global_step=best_step)
            writer_test.flush()
            print_log("TEST @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v), file=logger)

        if best_acc > 0.:
            print_log("Load best model without SWA  |  ACC={:.5f} form epoch={}".format(best_acc, best_epoch), file=logger)
            model_to_restore = get_best_model(log_dir, model='best_model')
            if model_to_restore is not None:
                best_saver.restore(sess, model_to_restore)
            else:
                print_log("Impossible to load best model .... ", file=logger)

            fit_bn_statistics(epoch, swa=False)
            sess.run(local_init_op)
            sampleNums = len(premises_np_test)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums), desc='TEST'):
                batch_premises_np_test = premises_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_test = premises_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_test = hypothesis_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_test = hypothesis_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_test = labels_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]

                feed_dict = {model.premise: batch_premises_np_test,
                             model.premise_mask: batch_premises_mask_test,
                             model.hypothesis: batch_hypothesis_np_test,
                             model.hypothesis_mask: batch_hypothesis_mask_test,
                             model.y: batch_labels_test,
                             model.is_training: False,
                             model.use_moving_statistics: False,
                             model.dropout_keep_prob: 1.0}
                sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)
            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
            writer_test.add_summary(s, global_step=best_step)
            writer_test.flush()
            print_log("TEST @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v), file=logger)

        if best_acc_swa > 0 and arg.use_swa:
            print_log("Load best model with SWA  |  ACC={:.5f} form epoch={}".format(best_acc_swa, best_epoch_swa), file=logger)
            model_to_restore = get_best_model(log_dir, model='best_model_swa')
            if model_to_restore is not None:
                # regular weights are already set to SWA weights ... no need to run 'retrieve_swa_weights' op.
                # and BN statistics are already set correctly
                best_saver_swa.restore(sess, model_to_restore)
            else:
                print_log("Impossible to load best model .... ", file=logger)

            fit_bn_statistics(epoch, swa=False)
            sess.run(local_init_op)
            sampleNums = len(premises_np_test)
            batchNums = int((sampleNums - 1) / arg.batch_size)
            for i in tqdm(range(batchNums), desc='TEST'):
                batch_premises_np_test = premises_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_premises_mask_test = premises_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_np_test = hypothesis_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_hypothesis_mask_test = hypothesis_mask_test[i * arg.batch_size: (i + 1) * arg.batch_size]
                batch_labels_test = labels_np_test[i * arg.batch_size: (i + 1) * arg.batch_size]

                feed_dict = {model.premise: batch_premises_np_test,
                             model.premise_mask: batch_premises_mask_test,
                             model.hypothesis: batch_hypothesis_np_test,
                             model.hypothesis_mask: batch_hypothesis_mask_test,
                             model.y: batch_labels_test,
                             model.is_training: False,
                             model.use_moving_statistics: False,
                             model.dropout_keep_prob: 1.0}
                sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)
            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
            writer_test.add_summary(s, global_step=best_step)
            writer_test.flush()
            print_log("TEST @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v), file=logger)

        writer_train.close()
        writer_val.close()
        writer_val_bn.close()
        writer_test.close()
        if arg.use_swa:
            writer_val_swa.close()
            writer_test_swa.close()

        sess.close()


if __name__ == '__main__':
    main()


