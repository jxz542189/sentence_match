import csv
import pandas as pd
import os
from zhconv import convert
import codecs
import re
import pkuseg
import tensorflow as tf
import numpy as np


path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(path, 'data')
seg = pkuseg.pkuseg()


def read_csv(file_name, sep='\t', index_col=0, N=None, head=True, header=None):
    csv_data = pd.read_csv(os.path.join(data_path, file_name), sep=sep, header=header, index_col=index_col)
    if N == None:
        return csv_data
    else:
        if head:
            return csv_data.head(N)
        else:
            return csv_data.tail(N)


def get_new_data_by_zhconv(file_name, column_names=None, output_filename="atec_new.csv"):
    with codecs.open(os.path.join(data_path, file_name)) as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_words = []
            words = line.split('\t')
            for word in words:
                temp_word = convert(word, 'zh-cn')
                # temp_word = re.sub('\\*\\*\\*', 'unknown',temp_word)
                new_words.append(temp_word)

            new_lines.append(new_words)
    list_write_csv(output_filename, new_lines, column_names=column_names)
    # with codecs.open(os.path.join(data_path, output_filename), 'w')  as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerows(new_lines)


def list_write_csv(csv_filename, unprocessed_list, column_names):
    csv_path = os.path.join(data_path, csv_filename)
    res = pd.DataFrame(columns=column_names, data=unprocessed_list)
    res.to_csv(csv_path, encoding="utf-8", index=False, sep="\t")
    

def get_each_class_number(file_name, sep='\t', index_col=0, N=None, head=True, header=0):
    csv_data = read_csv(file_name, sep=sep, index_col=index_col, N=N, header=header)
    return dict(csv_data['label'].value_counts())    #{0: 83792, 1: 18685}


def build_dataset(sess, x_train_np,
                  x_val_np,
                  x_test_np,
                  y_train_np,
                  y_val_np,
                  y_test_np,
                  batch_size=32,
                  buffer_size=40000,
                  emb_size=100,
                  sentence_len=30):
    with tf.variable_scope('DATA'):
        with tf.name_scope('dataset_placeholders'):
            x_train_tf = tf.placeholder(shape=[None, sentence_len, emb_size], dtype=tf.float32, name='X_train')
            x_val_tf = tf.placeholder(shape=[None, sentence_len, emb_size], dtype=tf.float32, name='X_val')
            x_test_tf = tf.placeholder(shape=[None, sentence_len, emb_size], dtype=tf.float32, name='X_test')

            y_train_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_train')
            y_val_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_val')
            y_test_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_test')

        with tf.name_scope('dataset_train'):
            dataset_train = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
            dataset_train = dataset_train.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            dataset_train = dataset_train.repeat()
            dataset_train = dataset_train.batch(batch_size=batch_size)

        with tf.name_scope('dataset_val'):
            dataset_val = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val_tf))
            dataset_val = dataset_val.batch(batch_size=batch_size)
            dataset_val = dataset_val.repeat()

        with tf.name_scope('dataset_test'):
            dataset_test = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test_tf))
            dataset_test = dataset_test.batch(batch_size=batch_size)
            dataset_test = dataset_test.repeat()

        with tf.name_scope('iterators'):
            handle = tf.placeholder(name="handle", shape=[], dtype=tf.string)
            iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types,
                                                           dataset_train.output_shapes)
            batch_x, batch_y = iterator.get_next()

            iterator_train = dataset_train.make_initializable_iterator()
            iterator_val = dataset_val.make_initializable_iterator()
            iterator_test = dataset_test.make_initializable_iterator()

        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())
        handle_test = sess.run(iterator_test.string_handle())

        print('...initialize datasets...')
        sess.run(iterator_train.initializer, feed_dict={x_train_tf: x_train_np, y_train_tf: y_train_np})
        sess.run(iterator_val.initializer, feed_dict={x_val_tf: x_val_np, y_val_tf: y_val_np})
        sess.run(iterator_test.initializer, feed_dict={x_test_tf: x_test_np, y_test_tf: y_test_np})

    return batch_x, batch_y, handle, handle_train, handle_val, handle_test


def get_data(file_name):
    csv_path = os.path.join(data_path, file_name)
    premises = []
    hypothesis = []
    labels = []
    with codecs.open(csv_path) as csv_file:
        csv_file.readline()
        for line in csv_file:
            sentences = line.split('\t')
            if len(sentences) == 4:
                sentence_1 = sentences[1]
                sentence_2 = sentences[2]
                label = sentences[3]
                premises.append(sentence_1)
                hypothesis.append(sentence_2)
                labels.append(label)
    return premises, hypothesis, labels


def get_words_id(file_name, output_filename='words.txt'):
    premises, hypothesis, labels = get_data(file_name)
    words_set = set()
    for line in premises:
        words = seg.cut(line)
        for word in words:
            words_set.add(word)
    for line in hypothesis:
        words = seg.cut(line)
        for word in words:
            words_set.add(word)
    words_id = {}
    words_id['unknow'] = 0
    i = 1
    with codecs.open(os.path.join(data_path, output_filename), 'w') as f:
        for word in words_set:
            words_id[word] = i
            i += 1
            f.write(word+'\n')
    return words_id


def get_text(file_name, output_filename="text.txt"):
    premises, hypothesis, labels = get_data(file_name)
    res = []
    for line in premises:
        res.append(" ".join(seg.cut(line)) + "\n")
    for line in hypothesis:
        res.append(" ".join(seg.cut(line)) + "\n")
    with codecs.open(os.path.join(data_path, output_filename), 'w') as f:
        f.writelines(res)


def get_word_to_vec(file_name):
    word_to_ids = {}
    word_to_ids['unknow'] = 0
    id_to_vec = {}
    i = 1

    with codecs.open(os.path.join(data_path, file_name)) as f:
        emb_size = int(f.readline().split()[1])
        id_to_vec[0] = np.random.rand(emb_size)
        txt = f.readline()
        while txt:
            if txt != '\r\n':
                word_and_vec = txt.split()
                word_to_ids[word_and_vec[0]] = i
                id_to_vec[i] = np.array(word_and_vec[1:], dtype=np.float32)
            txt = f.readline()
            i += 1
    return word_to_ids, id_to_vec


def sentence_to_ids(sentence, word_to_ids, sentence_len=30):
    words = seg.cut(sentence)
    words_id = []
    for word in words:
        if word in word_to_ids:
            words_id.append(word_to_ids[word])
        else:
            words_id.append(0)
    true_len = len(words) if len(words) <= sentence_len else sentence_len
    words_id = words_id[:sentence_len] if len(words_id) >= sentence_len else words_id + [0] * (sentence_len - len(words_id))
    return words_id, true_len


def sentence_to_vec(sentence, word_to_ids, id_to_vec, sentence_len=30):
    words_id, true_len = sentence_to_ids(sentence, word_to_ids, sentence_len)
    vecs = []
    for id in words_id:
        vecs.append(id_to_vec[id])
    return np.array(vecs), true_len


def get_traindata_and_testdata(file_name, word_to_ids, id_to_vec, sentence_len=30):
    premises, hypothesis, labels = get_data(file_name)
    premises_np = []
    premises_len = []
    hypothesis_np = []
    hypothesis_len = []
    labels_np = []
    for line in premises:
        sentence_vec, true_len = sentence_to_vec(line, word_to_ids, id_to_vec, sentence_len)
        premises_np.append(sentence_vec)
        premises_len.append(true_len)
    for line in hypothesis:
        sentence_vec, true_len = sentence_to_vec(line, word_to_ids, id_to_vec, sentence_len)
        hypothesis_np.append(sentence_vec)
        hypothesis_len.append(true_len)
    for label in labels:
        label = re.sub('\n', '', label)
        label = re.sub('\\"', '', label)
        if int(label) == 0:
            labels_np.append(np.array([1, 0], dtype=np.int32))
        else:
            labels_np.append(np.array([0, 1], dtype=np.int32))
    return np.array(premises_np), np.array(premises_len), np.array(hypothesis_np), np.array(hypothesis_len), np.array(labels_np, dtype=np.int32)


def get_data_id(file_name, word_to_ids, sentence_len=30):
    premises, hypothesis, labels = get_data(file_name)
    premises_np = []
    premises_len = []
    hypothesis_np = []
    hypothesis_len = []
    labels_np = []
    for line in premises:
        words_id, true_len = sentence_to_ids(line, word_to_ids, sentence_len)
        premises_np.append(np.array(words_id, dtype=np.int32))
        premises_len.append(true_len)
    for line in hypothesis:
        words_id, true_len = sentence_to_ids(line, word_to_ids, sentence_len)
        hypothesis_np.append(np.array(words_id, dtype=np.int32))
        hypothesis_len.append(true_len)
    for label in labels:
        label = re.sub('\n', '', label)
        label = re.sub('\\"', '', label)
        if int(label) == 0:
            labels_np.append(np.array([1, 0], dtype=np.int32))
        else:
            labels_np.append(np.array([0, 1], dtype=np.int32))
    return np.array(premises_np), np.array(premises_len), np.array(hypothesis_np), np.array(hypothesis_len), np.array(labels_np, dtype=np.int32)


if __name__ == '__main__':
    word_to_ids, id_to_vec = get_word_to_vec(file_name="vec_size100_mincount5.txt")
    premises_np, premises_mask, hypothesis_np, hypothesis_mask, labels_np = get_traindata_and_testdata("atec_new.csv", word_to_ids, id_to_vec)
    print(premises_np.shape)


    # get_text(file_name="atec_new.csv")

    # words_id = get_words_id("atec_new.csv")
    # print(words_id['unknow'])
    # print(words_id['花呗'])


    # res = get_each_class_number("atec_new.csv", header=0)
    # print(res)
    # print(res[0])


    # column_names = ['index', 'sentence_1', 'sentence_2', 'label']
    # get_new_data_by_zhconv("atec_nlp_sim_train_all.csv", column_names=column_names)

    # temp_str = "我是***月***用的花呗什么时候还款"
    # print(seg.cut(re.sub('\\*\\*\\*', 'unknown',temp_str)))
    # print(seg.cut(temp_str))

    # get_new_data_by_zhconv("atec_nlp_sim_train_all.csv")

    # print(convert('我幹什麼不干你事。', 'zh-cn'))
    # get_new_data_by_zhconv("atec_nlp_sim_train_all.csv")

