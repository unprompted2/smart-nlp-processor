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
        print("Modifying outputs...")

        # Pre assigning the data_outputs array
        data_outputs = np.zeros((self.n_examples, self.n_steps, len(self.available_slots)), dtype=np.int32)
        # Initiating all the one hot vectors to the default vector corresponding to the 'Outside' slot
        # Outside string 'O' is part of the naming convention (Begin, Inside and Outside)for classification of words (Object, Source etc) in a sentence.
        # Ref: $ROS_WORKSPACE/mbot_natural_language_processing/mbot_nlu/ros/doc/pedro_thesis.pdf
        idx_outside = self.available_slots.index('O')
        data_outputs[:, :, idx_outside] = 1
        # Initiating progress bar
        bar = progressbar.ProgressBar(max_value=len(outputs), redirect_stdout=True, end=' ')
        # Index for line wise iteration
        v = 0
        # Process outputs
        for line in outputs:
            # Index for word wise iteration
            w = 0
            for output in line:
                # find slot if it exists in available slots list
                try:
                    idx_found = self.available_slots.index(output)
                    # print('index found is ' + str(idx_found))
                except ValueError:
                    raise Exception('Could not find this output = {}  in this sentence = {} in the available list of slots'.format(output, sentences[outputs.index(line)]))
                # modify array
                data_outputs[v][w][idx_outside] = 0
                data_outputs[v][w][idx_found] = 1
                w = w + 1
            # Incrementing line index
            v = v + 1
            # Progress bar update
            bar.update(v)
        # Progress bar finished
        bar.finish()

        # debug prining
        if debug==True:
            print('Sample output data')
            print('=========================================================')
            print('output labels are {}'.format(outputs[0:2]))
            print('[Vector]output labels are {}'.format(outputs_train[0:2]))
            print('=========================================================')

        return word_vectors, data_inputs, data_outputs, lengths


    def shuffle_n_batch(self, data_inputs, data_outputs, lengths):
        '''
        shuffles and splits the data inputs for each epoch. 8 batches are used for training and 1 batch is used for validation
        '''
        split_list_inputs = []
        split_list_outputs = []
        split_list_lengths = []
        # resamplining data
        data_inputs, data_outputs, lengths = resample(data_inputs, data_outputs, lengths, replace=self.resample_replace)
        # splitting to batches
        len_each_chunks = int(len(data_inputs)/self.number_of_batches)
        for i in range(self.number_of_batches):
            split_list_inputs.append(np.array(data_inputs[len_each_chunks*(i):len_each_chunks*(i+1)]))
            split_list_outputs.append(np.array(data_outputs[len_each_chunks*(i):len_each_chunks*(i+1)]))
            split_list_lengths.append(np.array(lengths[len_each_chunks*(i):len_each_chunks*(i+1)]))

        return split_list_inputs, split_list_outputs, split_list_lengths


    def recurrent_neural_network(self, rnn_inputs, sequence_lengths):
        '''
        defining nn layers (graph in TF terms)
        '''
        # def of 1 layer of lstm
        def cell(): return tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, forget_bias=self.forget_bias)
        # initializing weights and biases for the output layer
        num_layers = self.n_lstm_layers
        layer = {'weights': tf.Variable(tf.random_normal(shape=[self.batch_size, 2*self.rnn_size, self.n_classes]), name='WEIGHTS'),
                 'biases': tf.Variable(tf.random_normal(shape=[self.n_classes]), name='BIASES')}

        # histograms for TB
        if self.use_tensorboard:
            tf.summary.histogram('weights', layer['weights'])
            tf.summary.histogram('biases', layer['biases'])

        # n layer of lstm with rnn_size number of cells for both forward and backward hidden layers.
        with tf.device('/GPU