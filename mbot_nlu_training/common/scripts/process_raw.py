#!/usr/bin/env python

from __future__ import print_function
from termcolor import colored

class BulkdData2SimplePhrases(object):
  """This class contains methods to transform the bulk generated data from RoboCup command generator
  to simple sorted (Alphabetical) phrases. Lines with #, questions and answers, and duplicates are also removed.
  The final list of phrases (1 phrase per line) is saved in a text file named 'processed'. This script
  helps in collecting and cleaning data required for training the neural network.
  """
  def __init__(self):
    self.file_name_flag = False

  def open_and_readlines(self, file_name):
    # open file and load lines
    with open(file_name,'r') as file:
      contents = file.readlines()
    return contents

  def get_file_name(self):
    '''
    Get file name to process. If not entered correctly print more info and request again
    '''
    while not self.file_name_flag:
      file_name = raw_input(colored("\nPlease enter the relative path of file (w.r.t pwd) you want to process[sample.txt]\n",'green')) or "sample.txt"
      if '.txt' not in file_name:
        p