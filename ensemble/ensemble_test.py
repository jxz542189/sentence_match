import tensorflow as tf
import numpy as np

# 建立两个 graph
g1 = tf.Graph()
g2 = tf.Graph()

# 为每个 graph 建创建一个 session
sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)

X_1 = None
tst_1 = None
yhat_1 = None

X_2 = None
tst_2 = None
yhat_2 = None


def load_model(sess):
    """
        Loading the pre-trained model and parameters.
    """
    global X_1, tst_1, yhat_1
    with sess1.as_default():
        with sess1.graph.as_default():
            modelpath = r'F:/resnet/model/new0.25-0.35/'
            saver = tf.train.import_meta_graph(modelpath + 'model-10.meta')
            saver.restore(sess1, tf.train.latest_checkpoint(modelpath))
            graph = tf.get_default_graph()
            X_1 = graph.get_tensor_by_name("X:0")
            tst_1 = graph.get_tensor_by_name("tst:0")
            yhat_1 = graph.get_tensor_by_name("tanh:0")
            print('Successfully load the model_1!')


def load_model_2():
    """
        Loading the pre-trained model and parameters.
    """
    global X_2, tst_2, yhat_2
    with sess2.as_default():
        with sess2.graph.as_default():
            modelpath = r'F:/resnet/model/new0.25-0.352/'
            saver = tf.train.import_meta_graph(modelpath + 'model-10.meta')
            saver.restore(sess2, tf.train.latest_checkpoint(modelpath))
            graph = tf.get_default_graph()
            X_2 = graph.get_tensor_by_name("X:0")
            tst_2 = graph.get_tensor_by_name("tst:0")
            yhat_2 = graph.get_tensor_by_name("tanh:0")
            print('Successfully load the model_2!')


def test_1(txtdata):
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41, 3).
        Test a single axample.
        Arg:
                txtdata: Array in C.
        Returns:
            The normal of a face.
    """
    global X_1, tst_1, yhat_1
    data = np.array(txtdata)
    data = data.reshape(-1, 41, 41, 41, 3)
    output = sess1.run(yhat_1, feed_dict={X_1: data, tst_1: True})  # (100, 3)
    output = output.reshape(-1, 1)
    ret = output.tolist()
    return ret


def test_2(txtdata):
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41, 3).
        Test a single axample.
        Arg:
                txtdata: Array in C.
        Returns:
            The normal of a face.
    """
    global X_2, tst_2, yhat_2

    data = np.array(txtdata)
    data = data.reshape(-1, 41, 41, 41, 3)
    output = sess2.run(yhat_2, feed_dict={X_2: data, tst_2: True})  # (100, 3)
    output = output.reshape(-1, 1)
    ret = output.tolist()

    return ret