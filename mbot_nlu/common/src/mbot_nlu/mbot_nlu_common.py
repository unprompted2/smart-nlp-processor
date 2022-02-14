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
        # test file exis