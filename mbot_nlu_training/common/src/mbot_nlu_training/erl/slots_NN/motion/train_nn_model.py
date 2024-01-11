import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


q = 0.2
n_examples = 20000
n_epochs = 15
embedding_size = 250
n_steps = 15    # n de palavras na frase. vai ser preciso fazer padding e organizar os batches mais ou menos por tamanhos
n_classes = 5    # n de ações
batch_size = 1 #int ( q * n_examples)
rnn_size = 500   # n of lstm hidden units
learning_rate = 0.001


def import_data(n_examples, n_steps):

    with open('wordvectors', 'rb') as vectors_file: #TODO: load from mbot_nlu_training/common/src/mbot_nlu_training/erl/wikipedia_vectors
        word_vectors = pickle.load(vectors_file)
        word_vectors[0] = np.zeros(250)

    with open('dictionary', 'rb') as dict_file: #TODO: load from mbot_nlu_training/common/src/mbot_nlu_training/erl/wikipedia_vectors
        dictionary = pickle.load(dict_file)

    vocab_size = len(dictionary)

    with open('inputs_slot_filling', 'rb') as data_inputs_file:
        sentences = pickle.load(data_inputs_file)

        lengths = []
        i = 0
        for line in sentences:
            
            h = []

            for k in range(len(line)):
                try:
                    idx = dictionary[line[k]]
                except:
                    pass
                h.append(idx)
            lengths.append(len(line))
            for _ in range(len(line), n_steps):
                h.append(0)
            i += 1
            if i == 1:
                data_inputs = np.array(h)
            else:
                data_inputs = np.vstack([data_inputs, h])

            inputs_train = np.array(data_inputs[int(q*n_examples):])
            inputs_test = np.array(data_inputs[:int(q*n_examples)])
            lengths_train = lengths[int(q*n_examples):]
            lengths_test = lengths[: int(q*n_examples)]

    with open('outputs_slot_filling', 'rb') as data_outputs_file:
        outputs = pickle.load(data_outputs_file)

        v=0
        data_outputs = [[] for _ in range(n_examples)]
        
        for line in outputs:
            for output in line:
                if 'Bgoal' in output:                              
                    data_outputs[v].append([1,0,0,0,0])
                elif 'Igoal' in output:                              
                    data_outputs[v].append([0,1,0,0,0])            
                elif 'Bpath' in output:                                   
                    data_outputs[v].append([0,0,1,0,0])
                elif 'Ipath' in output:                             
                    data_outputs[v].append([0,0,0,1,0])              
                elif 'O' in output:                                         
                    data_outputs[v].append([0,0,0,0,1])
                else:
                    print('2')
            for _ in range(len(data_outputs[v]), n_steps):
                data_outputs[v].append([0,0,0,0,1])
            v+=1

        outputs_train = np.array(data_outputs[int(q*n_examples):])
        outputs_test = np.array(data_outputs[: int(q*n_examples)])

    return [vocab_size, n_examples, inputs_train, inputs_test, outputs_train, outputs_test, lengths_train, lengths_test, word_vectors]

def recurrent_neural_network(rnn_size, n_classes, rnn_inputs, n_steps):
    
    layer = {'weights': tf.Variable(tf.random_normal([batch_size , 2 * rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    with tf.device('/gpu:2'):

        lstm_cell_bw = rnn.LSTMCell(rnn_size)
        lstm_cell_fw = rnn.LSTMCell(rnn_size)

    (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)

    with tf.device('/gpu:2'):

        outputs = tf.concat((outputs_fw, outputs_bw), 2)

        output = tf.m