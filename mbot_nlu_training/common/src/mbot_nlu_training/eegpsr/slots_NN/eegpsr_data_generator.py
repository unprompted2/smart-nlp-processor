#!/usr/bin/env python3

import random
import yaml
import msgpack
import numpy as np
from sklearn.utils import resample

# load parameters from yaml
# ================================================================================================================
yaml_dict = yaml.load(open('../../../../../ros/config/config_mbot_nlu_training.yaml'))['slots_train']
random_state = eval(yaml_dict['resample_random_state'])

# params for balancing individual structures
# ================================================================================================================
# number of types of different structured sentences in each of intent classes
n_struct = {'go': (45, True), 'take': (52, True), 'find': (44, True), 'answer': (2, True), 'tell': (11, True), 'guide': (18, True), 'follow': (14, True), 'meet': (0, False)}

# number of samples per structe required enough to make balances data
n_samples_per_intent = int(yaml_dict['n_examples']/len([item for item in n_struct.keys() if n_struct[item][0]!=0 and n_struct[item][1]]))
# data slider. bigger the value, bigger the number of sentences with complex structures(eg: grasp to mia at the kitchen the bottle from the bed room)
# but bigger the repeatation of sentences with smaller structure(eg: go to the kitchen)
data_slider = yaml_dict['data_slider']

# data for creating sentences eg: [names, objects]
# ================================================================================================================
# objects
# ================================================================================================================
objects_a = ['snack -Bobject-', 'cereals -Bobject bar -Iobject-', 'cookie -Bobject-', 'book -Bobject-', 'pen -Bobject-', 'notebook -Bobject-',
            'laptop -Bobject-', 'tablet -Bobject-', 'charger -Bobject-', 'pencil -Bobject-', 'peanut -Bobject-',
            'biscuit -Bobject-', 'candy -Bobject-', 'chocolate -Bobject bar -Iobject-', 'chewing -Bobject- gum -Iobject-',
            'chocolate -Bobject- egg -Iobject-', 'chocolate -Bobject- tablet -Iobject-', 'donuts -Bobject-', 'cake -Bobject-', 'pie -Bobject-',
            'peach -Bobject-', 'strawberry -Bobject-', 'blueberry -Bobject-', 'blackberry -Bobject-', 'burger -Bobject-', 'lemon -Bobject-',
            'banana -Bobject-', 'watermelon -Bobject-', 'pepper -Bobject-', 'pear -Bobject-', 'pizza -Bobject-', 'yogurt -Bobject-',
            'drink -Bobject-', 'beer -Bobject-', 'coke -Bobject-', 'sprite -Bobject-', 'sake -Bobject-', 'toothpaste -Bobject-',
            'cream -Bobject-', 'lotion -Bobject-', 'dryer -Bobject-', 'comb -Bobject-', 'towel -Bobject-', 'shampoo -Bobject-',
            'soap -Bobject-', 'cloth -Bobject-', 'sponge -Bobject-', 'toothbrush -Bobject-', 'container -Bobject-', 'glass -Bobject-',
            'can -Bobject-', 'bottle -Bobject-', 'fork -Bobject-', 'knife -Bobject-', 'bowl -Bobject-', 'tray -Bobject-', 'plate -Bobject-',
            'newspaper -Bobject-', 'magazine -Bobject-', 'kleenex -Bobject-', 'whiteboard -Bobject- cleaner -Iobject-']

objects_an = ['apple -Bobject-', 'almond -Bobject-', 'onion -Bobject-', 'orange -Bobject-']

objects_the = ['cookies -Bobject-', 'almonds -Bobject-', 'book -Bobject-', 'pen -Bobject-', 'notebook -Bobject-', 'laptop -Bobject-',
            'tablet -Bobject-', 'charger -Bobject-', 'pencil -Bobject-', 'chips -Bobject-', 'senbei -Bobject-', 'pringles -Bobject-',
            'peanuts -Bobject-', 'biscuits -Bobject-', 'crackers -Bobject-', 'candies -Bobject-', 'chocolate -Bobject bar -Iobject-',
            'manju -Bobject-', 'mints -Bobject-', 'chewing -Bobject- gums -Iobject-', 'chocolate -Bobject- egg -Iobject-',
            'chocolate -Bobject- tablet -Iobject-', 'donuts -Bobject-', 'cake -Bobject-', 'pie -Bobject-', 'food -Bobject-',
            'peach -Bobject-', 'strawberries -Bobject-', 'grapes -Bobject-', 'blueberries -Bobject-', 'blackberries -Bobject-',
            'salt -Bobject-', 'sugar -Bobject-', 'bread -Bobject-', 'cheese -Bobject-', 'ham -Bobject-', 'burger -Bobject-', 'ham -Bobject- burger -Iobject-',
            'lemon -Bobject-', 'onion -Bobject-', 'lemons -Bobject-', 'apples -Bobject-', 'onions -Bobject-', 'orange -Bobject-', 'oranges -Bobject-',
            'peaches -Bobject-', 'banana -Bobject-', 'bananas -Bobject-', 'noodles -Bobject-', 'apple -Bobject-', 'paprika -Bobject-',
            'watermelon -Bobject-', 'sushi -Bobject-', 'pepper -Bobject-', 'pear -Bobject-', 'pizza -Bobject-', 'yogurt -Bobject-',
            'drink -Bobject-', 'milk -Bobject-', 'juice -Bobject-', 'coffee -Bobject-', 'hot -Bobject- chocolate', 'whisky -Bobject-',
            'rum -Bobject-', 'vodka -Bobject-', 'cider -Bobject-', 'lemonade -Bobject-', 'tea -Bobject-', 'water -Bobject-', 'beer -Bobject-',
            'coke -Bobject-', 'sprite -Bobject-', 'wine -Bobject-', 'sake -Bobject-', 'toiletries -Bobject-', 'toothpaste -Bobject-',
            'cream -Bobject-', 'lotion -Bobject-', 'dryer -Bobject-', 'comb -Bobject-', 'towel -Bobject-', 'shampoo -Bobject-', 'soap -Bobject-',
            'cloth -Bobject-', 'sponge -Bobject-', 'toilet -Bobject- paper -Iobject-', 'toothbrush -Bobject-', 'container -Bobject-', 'containers -Bobject-',
            'glass -Bobject-', 'can -Bobject-', 'bottle -Bobject-', 'fork -Bobject-', 'knife -Bobject-', 'bowl -Bobject-', 'tray -Bobject-',
            'plate -Bobject-', 'newspaper -Bobject-', 'magazine -Bobject-', 'rice -Bobject-','kleenex -Bobject-',
            'whiteboard -Bobject- cleaner -Iobject-', 'cup -Bobject-', 'big -Bobject- dish -Iobject-', 'choco -Bobject- flakes -Iobject-']

objects_some = ['snacks -Bobject-', 'cookies -Bobject-', 'almonds -Bobject-', 'books -Bobject-', 'pens -Bobject-', 'chips -Bobject-',
            'pringles -Bobject-', 'magazines -Bobject-', 'newspapers -Bobject-', 'peanuts -Bobject-', 'biscuits -Bobject-', 'crackers -Bobject-',
            'candies -Bobject-', 'mints -Bobject-', 'chewing -Bobject- gums -Iobject-', 'donuts -Bobject-', 'cake -Bobject-', 'pie -Bobject-',
            'food -Bobject-', 'strawberries -Bobject-', 'grapes -Bobject-', 'blueberries -Bobject-', 'blackberries -Bobject-', 'salt -Bobject-',
            'sugar -Bobject-', 'bread -Bobject-', 'cheese -Bobject-', 'ham -Bobject-', 'lemons -Bobject-', 'apples -Bobject-',
            'onions -Bobject-', 'oranges -Bobject-', 'peaches -Bobject-', 'bananas -Bobject-', 'noodles -Bobject-', 'paprika -Bobject-',
            'watermelon -Bobject-', 'sushi -Bobject-', 'pepper -Bobject-', 'pizza -Bobject-', 'yogurt -Bobject-', 'drink -Bobject-',
            'milk -Bobject-', 'juice -Bobject-', 'coffee -Bobject-', 'hot -Bobject- chocolate -Iobject-',
            'whisky -Bobject-', 'rum -Bobject-', 'vodka -Bobject-', 'cider -Bobject-', 'lemonade -Bobject-', 'tea -Bobject-', 'water -Bobject-',
            'beer -Bobject-', 'coke -Bobject-', 'sprite -Bobject-', 'wine -Bobject-', 'sake -Bobject-', 'toilet -Bobject- paper -Iobject-',
            'containers -Bobject-', 'glasses -Bobject-', 'cans -Bobject-', 'bottles -Bobject-', 'forks -Bobject-', 'knives -Bobject-',
            'bowls -Bobject-', 'trays -Bobject-', 'plates -Bobject-', 'lemon -Bobject-', 'rice -Bobject-', 'cups -Bobject-']

objects_a_piece_of = ['cake -Bobject-', 'pie -Bobject-', 'bread -Bobject-', 'cheese -Bobject-', 'ham -Bobject-', 'watermelon -Bobject-',
                     'sushi -Bobject-', 'pizza -Bobject-', 'apple -Bobject-', 'lemon -Bobject-']

objects_a_cup_of = ['milk -Bobject-', 'coffee -Bobject-', 'hot -Bobject- chocolate -Iobject-', 'cider -Bobject-', 'lemonade -Bobject-',
                    'tea -Bobject-', 'water -Bobject-', 'beer -Bobject-', 'juice -Bobject-', 'rice -Bobject-']

objects_a_can_of = ['red -Bobject balls -Iobject-', 'cider -Bobject-', 'iced -Bobject- tea -Iobject-', 'beer -Bobject-', 'coke -Bobject-',
                     'sprite -Bobject-', 'juice -Bobject-', 'kleenex -Bobject-']

objects_a_glass_of = ['milk -Bobject-', 'juice -Bobject-', 'coffee -Bobject-', 'hot -Bobject- chocolate -Iobject-', 'whisky -Bobject-',
                    'rum -Bobject-', 'vodka -Bobject-', 'cider -Bobject-', 'lemonade -Bobject-', 'iced -Bobject- tea -Iobject-',
                    'water -Bobject-', 'beer -Bobject-', 'coke -Bobject-', 'sprite -Bobject-', 'wine -Bobject-', 'sake -Bobject-']

objects_a_bottle_of = ['milk -Bobject-', 'juice -Bobject-', 'whisky -Bobject-', 'rum -Bobject-', 'vodka -Bobject-', 'cider -Bobject-',
                     'lemonade -Bobject-', 'iced -Bobject- tea -Iobject-', 'water -Bobject-', 'beer -Bobject-', 'coke -Bobject-',
                     'sprite -Bobject-', 'wine -Bobject-','sake -Bobject-', 'kleenex -Bobject-']

obj