'''
Created on 2018.3.4
@author  : jxz
'''
#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import sys


class ModelConfig():
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.arg = None
        self.__addArguments()
        self.__readConfig()

    def __addArguments(self):
        # training hyper-parameters
        self.__parser.add_argument('--num_epochs',
                                   '-ep',
                                   default=300,
                                   type=int,
                                   help='Number of epochs')
        self.__parser.add_argument('--batch_size',
                                   '-bs',
                                   default=32,
                                   type=int,
                                   help='Batch size')
        self.__parser.add_argument('--dropout_keep_prob',
                                   '-dkp',
                                   default=0.5,
                                   type=float,
                                   help='Dropout keep probability')
        self.__parser.add_argument('--clip_value',
                                   '-cl',
                                   default=10,
                                   type=float,
                                   help='Norm to clip training')
        self.__parser.add_argument('--learning_rate',
                                   '-lr',
                                   default=0.0004,
                                   type=float,
                                   help='Learning rate')
        self.__parser.add_argument('--l2',
                                   '-l2',
                                   default=0.0,
                                   type=float,
                                   help='L2 normalization constant')
        self.__parser.add_argument('--seq_length',
                                   '-sl',
                                   default=30,
                                   type=int,
                                   help='Max length of input sentence')
        self.__parser.add_argument('--optimizer',
                                   '-op',
                                   default='adam',
                                   choices=['adagrad', 'adadelta', 'adam', 'sgd', 'rmsprop', 'momentum', 'adamW', 'momentumW'],
                                   type=str,
                                   help='Optimizer algorithm')
        self.__parser.add_argument('--early_stop_step',
                                   '-ess',
                                   default=50000,
                                   type=int,
                                   help='Early stop condition')

        # embeddings hyper-parameters
        self.__parser.add_argument('--threshold',
                                   '-th',
                                   default=0,
                                   type=int,
                                   help='Cut off freq(word) < threshold in vocabulary')
        self.__parser.add_argument('--embedding_size',
                                   '-es',
                                   default=300,
                                   type=int,
                                   help='Word embedding size')
        self.__parser.add_argument('--embedding_normalize',
                                   '-en',
                                   default=1,
                                   type=int,
                                   help='Normalize word embeddings')

        # layers hyper-parameters
        self.__parser.add_argument('--hidden_size',
                                   '-hs',
                                   default=300,
                                   type=int,
                                   help='Hidden layer size')
        self.__parser.add_argument('--attention_size',
                                   '-as',
                                   default=300,
                                   type=int,
                                   help='Attention layer size')

        # report hyper-parameters
        self.__parser.add_argument('--eval_batch',
                                   '-eb',
                                   default=1000,
                                   type=int,
                                   help='Number of batches between performance reports')
        self.__parser.add_argument('--num_train_steps',
                                   '-nts',
                                   default=600000,
                                   type=int,
                                   help='number of train steps')

        self.__parser.add_argument('--num_warmup_steps',
                                   '-nws',
                                   default=10000,
                                   type=int,
                                   help='number of warmup steps')

        self.__parser.add_argument('--cycle_epochs',
                                   '-ce',
                                   default=5,
                                   type=int,
                                   help='number epochs for each cycle')
        self.__parser.add_argument('--cycle_mode',
                                   '-cm',
                                   type=str,
                                   default="triangular",
                                   help='mode must be "triangular", "triangular2", or "exp_range"')
        self.__parser.add_argument('--opt', dest='opt', type=str, default='momentumW')  # adam, adamW, momentum, momentumW
        self.__parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
        self.__parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
        self.__parser.add_argument('--weight_decay_on', dest='weight_decay_on', type=str, default='all')  # all or kernels

        self.__parser.add_argument('--use_swa', dest='use_swa', type=int, default=1)
        self.__parser.add_argument('--epochs_before_swa', dest='epochs_before_swa', type=int, default=5)
        self.__parser.add_argument('--strategy_lr', dest='strategy_lr', type=str, default='swa')  # constant, swa
        self.__parser.add_argument('--cycle_length', dest='cycle_length', type=int, default=1)  # in epochs

        self.__parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.01)
        self.__parser.add_argument('--alpha1_lr', dest='alpha1_lr', type=float, default=0.01)
        self.__parser.add_argument('--alpha2_lr', dest='alpha2_lr', type=float, default=0.0001)
        # IO path
        ## embeddings
        self.__parser.add_argument('--vocab_path',
                                   '-vp',
                                   default='./SNLI/clean data/vocab.txt',
                                   type=str,
                                   help='Vocabulary file')
        self.__parser.add_argument('--embedding_path',
                                   '-embp',
                                   default='./SNLI/clean data/embeddings.pkl',
                                   type=str,
                                   help='Pre-trained word embeddings path')

        ## dataset
        self.__parser.add_argument('--trainset_path',
                                   '-trp',
                                   default='./SNLI/clean data/train.txt',
                                   type=str,
                                   help='Training set path')
        self.__parser.add_argument('--devset_path',
                                   '-dp',
                                   default='./SNLI/clean data/dev.txt',
                                   type=str,
                                   help='Validation set path')
        self.__parser.add_argument('--testset_path',
                                   '-tep',
                                   default='./SNLI/clean data/test.txt',
                                   type=str,
                                   help='Testing set path')

        ## reports
        self.__parser.add_argument('--save_path',
                                   '-sp',
                                   default='./model/checkpoint',
                                   type=str,
                                   help='Directory to save checkpoint')
        self.__parser.add_argument('--best_path',
                                   '-bp',
                                   default='./model/bestval',
                                   type=str,
                                   help='Directory to save the best model')
        self.__parser.add_argument('--log_path',
                                   '-lp',
                                   type=str,
                                   help='Log path')

        ## config
        self.__parser.add_argument('--config_path',
                                   '-cp',
                                   default='./ESIM_model/config/config.yaml',
                                   type=str,
                                   help='Config path')

        self.__parser.add_argument("--tfboard_path",
                                   '-board',
                                   default="./tensorboard",
                                   type=str,
                                   help='Tensorboard path')


    # read config information from config file
    def __readConfig(self):
        arg = self.__parser.parse_args()
        with open(arg.config_path) as conf:
            config_dict = yaml.load(conf)
            for key, value in config_dict.items():
                sys.argv.append('--' + key)
                sys.argv.append(str(value))
        self.arg = self.__parser.parse_args()

    # print config information
    def print_info(self):
        arg_dict = vars(self.arg)
        print('-' * 20 + ' Config Information ' + '-' * 20)
        for key, value in arg_dict.items():
            print('%-12s : %s' % (key, value))
        print('-' * 60)