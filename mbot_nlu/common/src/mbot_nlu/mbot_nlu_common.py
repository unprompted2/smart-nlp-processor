#!/usr/bin/env python

'''
GPSR General Purpose Service Robot Natural Language Understanding api
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import msgpack
import threading
import time
import os
import sys
import yaml
import inspect

# load list of available intents from configuration file, add to list if your training data has more intents.
available_intents = yaml.load(open(str(os.path.dirname(os.path.realpath(__file__))) + '/../../../../mbot_nlu_training/ros/config/config_mbot_nlu_training.yaml'))['test_params']['available_intents']


class NaturalLanguageUnderstanding(object):

    '''
    functions for natural language uderstanding
    input text, output intention (action) and slot (arguments)
    to be able to communicate with a robot in a natural way
    '''
    def __init__(self, classifier_path, wikipedia_vectors_path, debug=False):
        # classifier path
        self.base_path = classifier_path

        # whether to print verbose info
        self.debug = debug

        # to store the found intention and slots
        self.intention_found = None
        self.slot_found = None

        # if classifier is pedro_gpsr, according dictionary should be used
        dic_name = 'pedro_dictionary' if 'pedro_gpsr' in classifier_path else 'dictionary'
        # test file existance, wikipedia vectors is required
        if os.path.isfile(wikipedia_vectors_path + '/' + dic_name):
            print('Wikipedia dictionary file was found.. proceed')
        else:
            print('\033[91m' + '[ERROR] [mbot_nlu_common] Wikipedia dictionary file not found,' + '\033[0m')
            sys.exit()
        # load serialized object wikipedia dictionary
        with open(wikipedia_vectors_path + '/' + dic_name, 'rb') as dict_file:
            self.dictionary = msgpack.load(dict_file, raw=False)

        # check if user has requested verbose debug info
        if not self.debug:
            # Disable Tensorflow debugging information
            tf.logging.set_verbosity(tf.logging.ERROR)

        # print available intents if debug
        if debug: print('available intents = {}'.format(self.available_intents))


    def initialize_session(self):
        '''
        method to initialize session, graph and meta data from pretrained intent and slot classifiers
        '''

        # sanity check intent
        intent_initialization_var = ['self.intent_sess', 'self.x_intent', 'self.y_intent', 'self.sequence_length_intent']
        # if classifier exists or not
        classifier_check = True
        # check if the essential intent variables has been declared
        proceed_status = [True if var not in locals() else False for var in intent_initialization_var]

        # configuring gpu use and limiting memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if proceed_status.count(True)==len(proceed_status):
            print('Initiating intent classifier session... ', end='')
            # defining intent graph (graph is wrt terminology in TF)
            intent_graph = tf.Graph()
            # initiating tf session with intent graph
            self.intent_sess = tf.Session(graph=intent_graph, config=config)
            # setting intent_graph as default
            with intent_graph.as_default():
                try:
                    # restoring pretrained graph
                    saver_intent = tf.train.import_meta_graph(self.base_path + '/intent/actions_mydata_3.ckpt.meta', clear_devices=True)
                    # restoring meta data (contains variable data such as weights and biases)
                    saver_intent.restore(self.intent_sess, self.base_path + '/intent/actions_mydata_3.ckpt')
                    # assigning input tensor variable (1/2)
                    self.x_intent = intent_graph.get_tensor_by_name("input_placeholder:0")
                    # assigning input tensor variable (2/2)
                    self.sequence_length_intent = intent_graph.get_tensor_by_name("inputs_length:0")
                    # assigning output tensor variable
                    self.y_intent = intent_graph.get_tensor_by_name("Gather:0")
                except IOError:
                    print('\n\nIntention classifier missing!, are you sure you have downloaded them using the classifier setup? (mbot_nlu_classifier/common/setup/download_classifiers.sh)\n\n')
            print('Done')
        else:
            print('All the tf variables for intent session has been already initiated, moving on to slots session...')
            pass

        # sanity check slots
        slot_initialization_var = ['self.slot_sess', 'self.x_slot', 'self.y_slot', 'self.sequence_length_slot']
        # check if the essential intent variables has been declared
        proceed_status = [True if var not in locals() else False for var in slot_initialization_var]

        if proceed_status.count(True)==len(proceed_status):
            print('Initiating slot classifier session... ', end='')
            # defining slot graph
            slot_graph = tf.Graph()
            # initiating tf session with slot graph
            self.slot_sess = tf.Session(graph=slot_graph, config=config)
            # setting slot_graph as default
            with slot_graph.as_default():
                try:
                    # restoring pretrained graph
     