#!/usr/bin/env python3

import os
import time
import yaml
import shutil
import msgpack
import msgpack_numpy
import numpy as np
import progressbar
import tensorflow as tf
from termcolor import colored
from sklearn.utils import resample
from tensorflow.contrib import rnn
start_time = time.time()


class slot_training_class():
    '''
    This class contains a set of methods used to train the slots classifier for mbot_nlu package.
    '''
    def __init__(self, yaml_dict):
        '''
        constructor
        '''
        self.available_slots = yaml_dict['available_slots']                            # list of available intents, add to list if your training data has more intents.
        self.n_steps = yaml_dict['n_steps']                                            # max n of words in the sentences
        self.n_epochs = yaml_dict['n_epochs']                                          # n epochs
        self.base_path = yaml_dict['base_path']                                        # path to dictionary and wordvector files
        self.rnn_size = yaml_dict['rnn_size']                                          # n of lstm cells in each layer
        self.n_lstm_layers = yaml_dict['n_lstm_layers']                                # n lstm layers
        self.n_examples = yaml_dict['n_examples']                                      # n of samples for training and validation
        self.embedding_size = yaml_dict['embedding_size']                              # embedding size of each wordvector
        self.forget_bias = yaml_dict['forget_bias']                                    # forget bias for lstm cell
        self.learning_rate = yaml_dict['learning_rate']                                # initial learning rate for adam optimizer
        self.loss_lower_limit = yaml_dict['loss_lower_limit']                          # used to stop the training when the loss is below this value
        self.debug = yaml_dict['debug']                                                # debug, will give lot of info about the execution if True
        self.n_parallel_iterations_bi_rnn = yaml_dict['n_parallel_iterations_bi_rnn']  # The number of iterations to run in parallel...
                                                                                       # Those operations which do not have any temporal dependency and can be run in parallel...
                                                                                       # will be. This parameter trades off time for space. default=32
        self.train_method = yaml_dict['train_method']                                  # key to activate old and the new method,
                                                                                       # old = using tf placeholder with feed_dict, new = tf data iteration
        self.use_tensorboard = yaml_dict['use_tensorboard']                            # to use TB or not
        self.n_classes = len(self.available_slots)                                     # n of slots
        self.resample_replace = yaml_dict['resample_replace']                          # to use data duplication (randomly) while shuffling and batching

        # parameters for old method
        if self.train_method=='old':
            self.batch_size = yaml_dict['old_method']['batch_size']
            self.number_of_batches = yaml_dict['old_method']['number_of_batches']     # n batches
            self.number_of_validation_batches = yaml_dict['old_method']['number_of_validation_batches'] # n validation batches
            self.n_inputs_per_step = int(self.n_examples/self.number_of_batches)

        # parameters for new method
        if self.train_method=='new':
            self.batch_size = yaml_dict['new_method']['batch_size'] # batch size
            self.q = yaml_dict['new_method']['q']                   # Percentage of sample used for testing after each training session
            self.prefetch_buffer = self.batch_size                  # prefetching inputs to memory
            self.shuffle_buffer = self.n_examples                   # input data shuffle. Value > n_samples gives uniform shuffle
            self.repeat_dataset = self.n_epochs                     # repeat the dataset n number of times
            self.steps = int(int((1-self.q)*self.n_examples)/self.batch_size) # n steps per epoch (experimental)

        # printing parameters before training
        print('Training method = {}'.format(self.train_method))

    def import_data(self, debug=False):
        '''
        method for importing and processing input data
