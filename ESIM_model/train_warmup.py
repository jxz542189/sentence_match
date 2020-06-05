from ESIM_model.esim_model_by_warmup import ESIM
import os
from termcolor import colored
from datetime import datetime
from ESIM_model import Config
from ESIM_model.Utils import *
from utils.data_util import *
from utils.log import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import sys
import logging

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


logger = set_logger(colored('oredict', 'yellow'), False)


tf.reset_default_graph()

basedir = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(os.path.dirname(basedir), 'data')


def feed_data(model, premise, premise_mask, hypothesis,
              hypothesis_mask, y_batch, dropout_keep_prob):
    feed_dict = {
        model.premise: premise,
        model.premise_mask: premise_mask,
        model.hypothesis: hypothesis,
        model.hypothesis_mask: hypothesis_mask,
        model.y: y_batch,
        model.dropout_keep_prob: dropout_keep_prob
    }
    return feed_dict


def evaluate(sess, model, premise, premise_mask, hypothesis, hypothesis_mask, y):
    data_nums = len(premise)
    batchNums = int(data_nums // int(arg.batch_size))
    total_loss, total_acc = 0.0, 0.0
    for i in range(batchNums):
        batch_premises_np_test = premise[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_premises_mask_test = premise_mask[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_np_test = hypothesis[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_mask_test = hypothesis_mask[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_labels_test = y[i * arg.batch_size: (i + 1) * arg.batch_size]
        feed_dict = feed_data(model, batch_premises_np_test, batch_premises_mask_test, batch_hypothesis_np_test,
                              batch_hypothesis_mask_test, batch_labels_test,
                              arg.dropout_keep_prob)
        batch_loss, batch_acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += batch_loss
        total_acc += batch_acc
    return total_loss / batchNums, total_acc / batchNums



def train(model):
    logger.info('Loading training and validation data ...')
    word_to_ids = get_word_to_id(file_name=os.path.join(data_path, 'vocab.txt'))

    premises_np, premises_mask, hypothesis_np, hypothesis_mask, labels_np = get_data_id(
        os.path.join(data_path, "atec_nlp_sim_train_all.csv"), word_to_ids, arg.seq_length)

    premises_np_train, premises_np_test = train_test_split(premises_np, random_state=1234)
    premises_mask_train, premises_mask_test = train_test_split(premises_mask, random_state=1234)
    hypothesis_np_train, hypothesis_np_test = train_test_split(hypothesis_np, random_state=1234)
    hypothesis_mask_train, hypothesis_mask_test = train_test_split(hypothesis_mask, random_state=1234)
    labels_np_train, labels_np_test = train_test_split(labels_np, random_state=1234)

    data_nums = len(premises_np_train)
    saver = tf.train.Saver(max_to_keep=5)
    save_file_dir, save_file_name = os.path.split(arg.save_path)
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    logger.info('Configuring TensorBoard and Saver ...')
    if not os.path.exists(arg.tfboard_path):
        os.makedirs(arg.tfboard_path)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(arg.tfboard_path)
    # word_to_ids, id_to_vec = get_word_to_vec(file_name="vec_size100_mincount5.txt")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_parameters = count_parameters()
    logger.info('Total trainable parameters : {}'.format(total_parameters))

    logger.info('Start training and evaluating ...',)
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved_batch = 0
    isEarlyStop = False
    for epoch in range(arg.num_epochs):
        logger.info("Epoch : {}".format(epoch + 1))
        sampleNums = len(premises_np_train)
        batchNums = int((sampleNums - 1) / arg.batch_size)

        indices = np.random.permutation(np.arange(sampleNums))
        premises_np_train = premises_np_train[indices]
        premises_mask_train = premises_mask_train[indices]
        hypothesis_np_train = hypothesis_np_train[indices]
        hypothesis_mask_train = hypothesis_mask_train[indices]
        labels_np_train = labels_np_train[indices]

        # batches = next_batch(premises_np_train, premises_mask_train, hypothesis_np_train, hypothesis_mask_train, labels_np_train)
        total_loss, total_acc = 0.0, 0.0
        for i in range(batchNums):
            batch_premises_np_train = premises_np_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_premises_mask_train = premises_mask_train[i * arg.batch_size: (i + 1) * arg.batch_size]
            batch_hypothesis_np_train = hypothesis_np_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_hypothesis_mask_train = hypothesis_mask_train[i * arg.batch_size: (i + 1) * arg.batch_size]
            batch_labels_train = labels_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]
            batch_nums = arg.batch_size
            feed_dict = feed_data(model, batch_premises_np_train, batch_premises_mask_train, batch_hypothesis_np_train,
                                  batch_hypothesis_mask_train, batch_labels_train,
                                  arg.dropout_keep_prob)
            _, batch_loss, batch_acc = sess.run([model.train, model.loss, model.acc], feed_dict=feed_dict)
            total_loss += batch_loss * batch_nums
            total_acc += batch_acc * batch_nums

            if total_batch % arg.eval_batch == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

                feed_dict[model.dropout_keep_prob] = 1.0
                loss_val, acc_val = evaluate(sess, model, premises_np_test,
                                             premises_mask_test,
                                             hypothesis_np_test,
                                             hypothesis_mask_test,
                                             labels_np_test)
                saver.save(sess=sess, save_path=arg.save_path + '_dev_loss_{:.4f}.ckpt'.format(loss_val))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved_batch = total_batch
                    saver.save(sess=sess, save_path=arg.best_path)
                    improved_flag = '*'
                else:
                    improved_flag = ''
                time_diff = get_time_diff(start_time)
                msg = 'Epoch : {0:>3}, Batch : {1:>8}, Train Batch Loss : {2:>6.2}, Train Batch Acc : {3:>6.2%}, Dev Loss : {4:>6.2}, Dev Acc : {5:>6.2%}, Time : {6} {7}'
                logger.info(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val, acc_val, time_diff,
                                     improved_flag))
            total_batch += 1
            if total_batch - last_improved_batch > arg.early_stop_step:
                logger.info('No optimization for a long time, auto-stopping ...')
                isEarlyStop = True
                break
        if isEarlyStop:
            break
        time_diff = get_time_diff(start_time)
        total_loss, total_acc = total_loss / data_nums, total_acc / data_nums
        msg = '** Epoch : {0:>2} finished, Train Loss : {1:>6.2}, Train Acc : {2:6.2%}, Time : {3}'
        logger.info(msg.format(epoch + 1, total_loss, total_acc, time_diff))


if __name__ == '__main__':
    config = Config.ModelConfig()
    arg = config.arg

    arg.n_classes = 2
    logger.info('CMD : python3 {0}'.format(' '.join(sys.argv)))
    logger.info('Training with following options :')
    word_to_ids = get_word_to_id(file_name=os.path.join(data_path, 'vocab.txt'))
    arg.n_vocab = len(word_to_ids)

    model = ESIM(arg.seq_length, arg.n_vocab, arg.embedding_size, arg.hidden_size, arg.attention_size, arg.n_classes,
                 arg.batch_size, arg.learning_rate, arg.optimizer, arg.l2, arg.clip_value, arg.num_train_steps, arg.num_warmup_steps)
    train(model)


