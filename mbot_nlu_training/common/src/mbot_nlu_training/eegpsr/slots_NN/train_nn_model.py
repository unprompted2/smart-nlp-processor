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
   