#!/usr/bin/env python3

import os
import sys
import time
import yaml
import unittest
import progressbar
sys.path.append(os.path.abspath('../'))
from src.mbot_nlu.mbot_nlu_common import NaturalLanguageUnderstanding

class MbotNluTest(unittest.TestCase):

    def setUp(self):
        '''
        Sets up the test fixture befor