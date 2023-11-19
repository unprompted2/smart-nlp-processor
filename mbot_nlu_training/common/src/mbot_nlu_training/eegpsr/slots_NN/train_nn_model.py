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
        '''
        # Importing pickled wordvectors, dictionary, inputs and labels
        with open(self.base_path + '/wordvectors', 'rb') as vectors_file:
            print("Importing wordvectors...", end=' ', flush=True)
            word_vectors = msgpack_numpy.load(vectors_file)
            print("Done")
        with open(self.base_path + '/dictionary', 'rb') as dict_file:
            print("Importing dictionary...", end=' ', flush=True)
            dictionary = msgpack.load(dict_file, raw=False)
            print("Done")
        with open('inputs_slot_filling', 'rb') as data_inputs_file:
            print("Importing inputs...", end=' ', flush=True)
            sentences = msgpack.load(data_inputs_file, raw=False)
            print("Done")
        with open('outputs_slot_filling', 'rb') as data_outputs_file:
            print("Importing labels...", end=' ', flush=True)
            outputs = msgpack.load(data_outputs_file, raw=False)
            print("Done")


        ########################################################################################################################
        # Processing inputs
        ########################################################################################################################
        print("Modifying input sentences...")

        # importing progressbar
        bar = progressbar.ProgressBar(max_value=len(sentences), redirect_stdout=True, end=' ')
        # preassigning the inputs variable for faster processing
        data_inputs = np.zeros((len(sentences), self.n_steps), dtype=np.int32)
        # initiating all the inputs to index of zero vector (zerowordvec_idx = dictionary['zerowordvec'])
        zerowordvec_idx = dictionary['zerowordvec']
        data_inputs[:, :] = zerowordvec_idx
        # Processing inputs
        lengths = np.zeros(len(sentences), dtype=np.int32)
        i = 0
        words_not_found_in_dic = []
        for line in sentences:
            # Initializing an empty list of Indexes
            h = []
            # Iterating each word in the line over the dictionary and appending the indexes to a list
            for k in range(len(line)):
                try:
                    idx = dictionary[line[k]]
                except:
                    idx = zerowordvec_idx
                    words_not_found_in_dic.append(line[k])
                # Appending the index(idx) of each word to the list h.
                h.append(idx)
            # appending the length of each line to the list lengths
            lengths[i] = len(line)
            # modify contents of the array
            data_inputs[i, :len(h)] = h
            # bar update
            bar.update(i)
            i = i + 1
        # bar finish
        bar.finish()
        # if words are not found in dictionary
        if len(words_not_found_in_dic)!=0:
            # rm duplicates
            words_not_found_in_dic = list(set(words_not_found_in_dic))
            # use this file to update most_common_words and to generate a new dictionary and wordvector
            with open('words_not_found_in_dic.txt','w') as f:
                for item in words_not_found_in_dic: f.write(item + '\n')
            print(colored('\nNo. of words not found in the dict = {}, pls. check words_not_found_in_dic file\n'.format(len(words_not_found_in_dic)),'red'), end='', flush=True)
        # if debug print input sample to check if the input pipeline is correct
        if debug:
            print('Sample input data')
            print('=========================================================')
            print('input sentences are {}'.format(sentences[0:2]))
            print('[Vector]input sentence are {}'.format(data_inputs[0:2]))
            print('=========================================================')

        ########################################################################################################################
        # Processing labels
        ########################################################################################################################
        print("Modifying outp