from DIIN_model.diin import MyModel
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import logging
import tensorflow as tf
from utils.data_util import get_word_to_vec, get_data_id_diin
from DIIN_model.Util import print_log, count_parameters, get_time_diff, print_args
from DIIN_model.util import parameters as params
import numpy as np
import time

tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(path, 'data')
dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_path = os.path.join(path, 'DIIN_model', "logs/log.{}".format(dt))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path)
logger = logging.getLogger(__name__)





def feed_data(model, premise, hypothesis,
              premise_pos, hypothesis_pos,
              premise_char, hypothesis_char,
              premise_exact_match, hypothesis_exact_match,
              y, is_train=True):
    feed_dict = {
        model.premise_x: premise,
        model.hypothesis_x: hypothesis,
        model.premise_pos: premise_pos,
        model.hypothesis_pos: hypothesis_pos,
        model.premise_char: premise_char,
        model.hypothesis_char: hypothesis_char,
        model.premise_exact_match: premise_exact_match,
        model.hypothesis_exact_match: hypothesis_exact_match,
        model.y: y,
        model.is_train: is_train
    }
    return feed_dict


def evaluate(sess, model, premise, hypothesis,
              premise_pos, hypothesis_pos,
              premise_char, hypothesis_char,
              premise_exact_match, hypothesis_exact_match,
              y):
    data_nums = len(premise)
    batchNums = int(data_nums // int(arg.batch_size))
    total_loss, total_acc = 0.0, 0.0
    for i in range(batchNums):
        batch_premises_np_train = premise[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_np_train = hypothesis[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_premise_pos_train = premise_pos[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_pos_train = hypothesis_pos[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_premise_char_train = premise_char[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_char_train = hypothesis_char[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_premise_exact_match_train = premise_exact_match[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_hypothesis_exact_match_train = hypothesis_exact_match[i * arg.batch_size: (i + 1) * arg.batch_size]
        batch_labels_train = y[i * arg.batch_size: (i + 1) * arg.batch_size]
        feed_dict = feed_data(model, batch_premises_np_train, batch_hypothesis_np_train,
                              batch_premise_pos_train, batch_hypothesis_pos_train,
                              batch_premise_char_train, batch_hypothesis_char_train,
                              batch_premise_exact_match_train, batch_hypothesis_exact_match_train,
                              batch_labels_train, is_train=False)
        _, batch_loss, batch_acc = sess.run([model.train_op, model.total_loss, model.acc], feed_dict=feed_dict)
        total_loss += batch_loss
        total_acc += batch_acc
    return total_loss / batchNums, total_acc / batchNums


def train(model, arg):
    print_log('Loading training and validation data ...', file=logger)
    word_to_ids, id_to_vec = get_word_to_vec(file_name="vec_size100_mincount5.txt")
    premises_np, premises_mask, hypothesis_np, \
    hypothesis_mask, premises_pos, hypothesis_pos, \
    premises_char, hypothesis_char, premise_exact_match, \
    hypothesis_exact_match, labels_np = get_data_id_diin("atec_new.csv", word_to_ids, arg.seq_length, arg.char_in_word_size)
    if not os.path.exists(os.path.join(data_path, 'premises_np_size100_mincount5.m')):
        joblib.dump(premises_np, os.path.join(data_path, 'premises_np_size100_mincount5.m'))
        joblib.dump(premises_mask, os.path.join(data_path, 'premises_mask_size100_mincount5.m'))
        joblib.dump(hypothesis_np, os.path.join(data_path, 'hypothesis_np_size100_mincount5.m'))
        joblib.dump(hypothesis_mask, os.path.join(data_path, 'hypothesis_mask_size100_mincount5.m'))
        joblib.dump(premises_pos, os.path.join(data_path, 'premise_pos_size100_mincount5.m'))
        joblib.dump(hypothesis_pos, os.path.join(data_path, 'hypothesis_pos_size100_mincount5.m'))
        joblib.dump(premises_char, os.path.join(data_path, 'premises_char_size100_mincount5.m'))
        joblib.dump(hypothesis_char, os.path.join(data_path, 'hypothesis_char_size100_mincount5.m'))
        joblib.dump(premise_exact_match, os.path.join(data_path, 'premise_exact_match_size100_mincount5.m'))
        joblib.dump(hypothesis_exact_match, os.path.join(data_path, 'hypothesis_exact_match_size100_mincount5.m'))
        joblib.dump(labels_np, os.path.join(data_path, 'labels_np_size100_mincount5.m'))
    else:
        premises_np = joblib.load(os.path.join(data_path, 'premises_np_size100_mincount5.m'))
        premises_mask = joblib.load(os.path.join(data_path, 'premises_mask_size100_mincount5.m'))
        hypothesis_np = joblib.load(os.path.join(data_path, 'hypothesis_np_size100_mincount5.m'))
        hypothesis_mask = joblib.load(os.path.join(data_path, 'hypothesis_mask_size100_mincount5.m'))
        premises_pos = joblib.load(os.path.join(data_path, 'premise_pos_size100_mincount5.m'))
        hypothesis_pos = joblib.load(os.path.join(data_path, 'hypothesis_pos_size100_mincount5.m'))
        premises_char = joblib.load(os.path.join(data_path, 'premises_char_size100_mincount5.m'))
        hypothesis_char = joblib.load(os.path.join(data_path, 'hypothesis_char_size100_mincount5.m'))
        premise_exact_match = joblib.load(os.path.join(data_path, 'premise_exact_match_size100_mincount5.m'))
        hypothesis_exact_match = joblib.load(os.path.join(data_path, 'hypothesis_exact_match_size100_mincount5.m'))
        labels_np = joblib.load(os.path.join(data_path, 'labels_np_size100_mincount5.m'))


    premises_np_train, premises_np_test = train_test_split(premises_np, random_state=1234)
    premises_mask_train, premises_mask_test = train_test_split(premises_mask, random_state=1234)
    hypothesis_np_train, hypothesis_np_test = train_test_split(hypothesis_np, random_state=1234)
    hypothesis_mask_train, hypothesis_mask_test = train_test_split(hypothesis_mask, random_state=1234)
    premises_pos_train, premise_pos_test = train_test_split(premises_pos, random_state=1234)
    hypothesis_pos_train, hypothesis_pos_test = train_test_split(hypothesis_pos, random_state=1234)
    premises_char_train, premises_char_test = train_test_split(premises_char, random_state=1234)
    hypothesis_char_train, hypothesis_char_test = train_test_split(hypothesis_char, random_state=1234)
    premise_exact_match_train, premise_exact_match_test = train_test_split(premise_exact_match, random_state=1234)
    hypothesis_exact_match_train, hypothesis_exact_match_test = train_test_split(hypothesis_exact_match, random_state=1234)
    labels_np_train, labels_np_test = train_test_split(labels_np, random_state=1234)

    data_nums = len(premises_np_train)
    saver = tf.train.Saver(max_to_keep=5)
    save_file_dir = os.path.join(path, 'DIIN_model', 'DIIN_model')
    save_file_name = 'model'
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    #
    print_log('Configuring TensorBoard and Saver ...', file=logger)
    tfboard_path = os.path.join(path, 'DIIN_model', 'tfboard_path')
    if not os.path.exists(tfboard_path):
        os.makedirs(tfboard_path)
    tf.summary.scalar('loss', model.total_cost)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(arg.tfboard_path)
    embeddings = []
    for i in range(len(id_to_vec)):
        embeddings.append(id_to_vec[i])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_parameters = count_parameters()
    print_log('Total trainable parameters : {}'.format(total_parameters), file=logger)

    print_log('Start training and evaluating ...', file=logger)
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved_batch = 0
    isEarlyStop = False
    for epoch in range(arg.num_epochs):
        print_log("Epoch : {}".format(epoch + 1), file=logger)
        sampleNums = len(premises_np_train)
        batchNums = int((sampleNums - 1) / arg.batch_size)

        indices = np.random.permutation(np.arange(sampleNums))
        premises_np_train = premises_np_train[indices]
        premises_mask_train = premises_mask_train[indices]
        hypothesis_np_train = hypothesis_np_train[indices]
        hypothesis_mask_train = hypothesis_mask_train[indices]
        premises_pos_train = premises_pos_train[indices]
        hypothesis_pos_train = hypothesis_pos_train[indices]
        premises_char_train = premises_char_train[indices]
        hypothesis_char_train = hypothesis_char_train[indices]
        premise_exact_match_train = premise_exact_match_train[indices]
        hypothesis_exact_match_train = hypothesis_exact_match_train[indices]
        labels_np_train = labels_np_train[indices]
        total_loss, total_acc = 0.0, 0.0
        for i in range(batchNums):
            batch_premises_np_train = premises_np_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_hypothesis_np_train = hypothesis_np_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_premises_pos_train = premises_pos_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_hypothesis_pos_train = hypothesis_pos_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_premises_char_train = premises_char_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_hypothesis_char_train = hypothesis_char_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_premise_exact_match_train = premise_exact_match_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_hypothesis_exact_match_train = hypothesis_exact_match_train[i * arg.batch_size: (i+1)*arg.batch_size]
            batch_labels_train = labels_np_train[i * arg.batch_size: (i + 1) * arg.batch_size]
            batch_nums = arg.batch_size
            feed_dict = feed_data(model, batch_premises_np_train,
                                  batch_hypothesis_np_train,
                                  batch_premises_pos_train,
                                  batch_hypothesis_pos_train,
                                  batch_premises_char_train,
                                  batch_hypothesis_char_train,
                                  batch_premise_exact_match_train,
                                  batch_hypothesis_exact_match_train,
                                  batch_labels_train)
            _, batch_loss, batch_acc = sess.run([model.train_op, model.total_cost, model.acc], feed_dict=feed_dict)
            total_loss += batch_loss * batch_nums
            total_acc += batch_acc * batch_nums

            if total_batch % arg.eval_batch == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

                loss_val, acc_val = evaluate(sess, model, premises_np_test,
                                             hypothesis_np_test,
                                             premise_pos_test,
                                             hypothesis_pos_test,
                                             premises_char_test,
                                             hypothesis_char_test,
                                             premise_exact_match_test,
                                             hypothesis_exact_match_test,
                                             labels_np_test)
                saver.save(sess=sess, save_path=os.path.join(save_file_dir, save_file_name) + '_dev_loss_{:.4f}.ckpt'.format(loss_val))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved_batch = total_batch
                    saver.save(sess=sess, save_path=os.path.join(save_file_dir, 'best_model.ckpt'))
                    improved_flag = '*'
                else:
                    improved_flag = ''
                time_diff = get_time_diff(start_time)
                msg = 'Epoch : {0:>3}, Batch : {1:>8}, Train Batch Loss : {2:>6.2}, Train Batch Acc : {3:>6.2%}, Dev Loss : {4:>6.2}, Dev Acc : {5:>6.2%}, Time : {6} {7}'
                print_log(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val, acc_val, time_diff,
                                     improved_flag), file=logger)
            total_batch += 1
            if total_batch - last_improved_batch > arg.early_stop_step:
                print_log('No optimization for a long time, auto-stopping ...', file = logger)
                isEarlyStop = True
                break
        if isEarlyStop:
            break
        time_diff = get_time_diff(start_time)
        total_loss, total_acc = total_loss / data_nums, total_acc / data_nums
        msg = '** Epoch : {0:>2} finished, Train Loss : {1:>6.2}, Train Acc : {2:6.2%}, Time : {3}'
        print_log(msg.format(epoch + 1, total_loss, total_acc, time_diff), file = logger)


if __name__ == '__main__':
    arg = params.args
    print_args(arg, logger)
    tok_mat = np.random.randn(arg.n_tokens, arg.emb_dim).astype(np.float32) / np.sqrt(arg.emb_dim)
    model = MyModel(arg, arg.seq_length, arg.emb_dim, arg.hidden_dim, True, embeddings=tok_mat, pred_size=2)
    train(model, arg)


