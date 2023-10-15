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

objects = list(set(objects_a + objects_the + objects_some + objects_an + objects_a_piece_of + objects_a_cup_of + objects_a_can_of + objects_a_bottle_of + objects_a_glass_of))

del objects_a, objects_some, objects_an, objects_a_piece_of, objects_a_cup_of, objects_a_can_of, objects_a_bottle_of, objects_a_glass_of

# ================================================================================================================
# locations
# ================================================================================================================
locations_on = ['nightstand -Blocation-', 'bookshelf -Blocation-', 'coffee -Blocation- table -Ilocation-', 'side -Blocation- table -Ilocation-',
                'kitchen -Blocation- table -Ilocation-', 'kitchen -Blocation- cabinet -Ilocation-', 'tv -Blocation- stand -Ilocation-',
                'sofa -Blocation-', 'couch -Blocation-', 'bedroom -Blocation- chair -Ilocation-', 'kitchen -Blocation- chair -Ilocation-',
                'living -Blocation- room -Ilocation- table -Ilocation-', 'center -Blocation- table -Ilocation-', 'drawer -Blocation-', 'desk -Blocation-',
                'cupboard -Blocation-', 'side -Blocation- shelf -Ilocation-', 'bookcase -Blocation-', 'dining -Blocation- table -Ilocation-',
                'fridge -Blocation-', 'counter -Blocation-', 'cabinet -Blocation-', 'table -Blocation-', 'bedchamber -Blocation-', 'chair -Blocation-',
                'fridge -Blocation- 2 -Ilocation-', 'counter -Blocation- 4 -Ilocation-', 'cabinet -Blocation- 2 -Ilocation-',
                'table -Blocation- 1 -Ilocation-', 'bedchamber -Blocation- 3 -Ilocation-', 'chair -Blocation- 8 -Ilocation-',
                'dryer -Blocation-', 'oven -Blocation-', 'rocking -Blocation- chair -Ilocation-', 'stove -Blocation-', 'television -Blocation-',
                'dressing -Blocation- table -Ilocation-', 'bench -Blocation-', 'futon -Blocation-', 'beanbag -Blocation-', 'stool -Blocation-',
                'dressing -Blocation- table -Ilocation- 1 -Ilocation-', 'bench -Blocation- 1 -Ilocation-', 'futon -Blocation- 1 -Ilocation-',
                'beanbag -Blocation- 1 -Ilocation-', 'stool -Blocation- 1 -Ilocation-',
                'sideboard -Blocation-', 'washing -Blocation- machine -Ilocation-', 'dishwasher -Blocation-']

locations_in = ['wardrobe -Blocation-', 'nightstand -Blocation-', 'bookshelf -Blocation-', 'dining -Blocation- room -Ilocation-', 'bedroom -Blocation-',
                'closet -Blocation-', 'living -Blocation- room -Ilocation-', 'bar -Blocation-', 'office -Blocation-', 'drawer -Blocation-',
                'kitchen -Blocation-', 'cupboard -Blocation-', 'side -Blocation- shelf -Ilocation-', 'fridge -Blocation-', 'corridor -Blocation-',
                'cabinet -Blocation-', 'bathroom -Blocation-', 'toilet -Blocation-', 'hall -Blocation-', 'hallway -Blocation-',
                'cabinet -Blocation- 1 -Ilocation-', 'bathroom -Blocation- 5 -Ilocation-', 'toilet -Blocation- 6 -Ilocation-', 'hall -Blocation- 4 -Ilocation-', 'hallway -Blocation- 2 -Ilocation-',
                'master -Blocation- bedroom -Ilocation- 3 -Ilocation-', 'dormitory -Blocation- room -Ilocation- 2 -Ilocation-', 'bedchamber -Blocation- 7 -Ilocation-', 'cellar -Blocation- 0 -Ilocation-',
                'master -Blocation- bedroom -Ilocation-', 'dormitory -Blocation- room -Ilocation-', 'bedchamber -Blocation-', 'cellar -Blocation-',
                'den -Blocation-', 'garage -Blocation-', 'playroom -Blocation-', 'porch -Blocation-', 'staircase -Blocation-',
                'sun -Blocation- room -Ilocation-', 'music -Blocation- room -Ilocation-', 'prayer -Blocation- room -Ilocation-',
                'utility -Blocation- room -Ilocation-', 'shed -Blocation-', 'basement -Blocation-', 'workshop -Blocation-',
                'ballroom -Blocation-', 'box -Blocation- room -Ilocation-', 'conservatory -Blocation-', 'drawing -Blocation- room -Ilocation-',
                'games -Blocation- room -Ilocation-', 'larder -Blocation-', 'library -Blocation-', 'parlour -Blocation-', 'guestroom -Blocation-',
                'crib -Blocation-', 'shower -Blocation-']

locations_at = ['wardrobe -Blocation-', 'nightstand -Blocation-', 'bookshelf -Blocation-', 'coffee -Blocation- table -Ilocation-',
                'side -Blocation- table -Ilocation-', 'kitchen -Blocation- table -Ilocation-', 'kitchen -Blocation- cabinet -Ilocation-',
                'bed -Blocation-', 'bedside -Blocation-', 'closet -Blocation-', 'tv -Blocation- stand -Ilocation-', 'sofa -Blocation-',
                'couch -Blocation-', 'bedroom -Blocation- chair -Ilocation-', 'kitchen -Blocation- chair -Ilocation-',
                'living -Blocation- room -Ilocation- table -Ilocation-', 'center -Blocation- table -Ilocation-', 'bar -Blocation-',
                'drawer -Blocation-', 'desk -Blocation-', 'cupboard -Blocation-', 'sink -Blocation-', 'side -Blocation- shelf -Ilocation-',
                'bookcase -Blocation-', 'dining -Blocation- table -Ilocation-', 'fridge -Blocation-', 'counter -Blocation-', 'door -Blocation-',
                'bookcase -Blocation- 0 -Ilocation-', 'dining -Blocation- table -Ilocation- 2 -Ilocation-', 'fridge -Blocation- 1 -Ilocation-',
                'counter -Blocation- 6 -Ilocation-', 'door -Blocation- 4 -Ilocation-', 'cabinet -Blocation-', 'table -Blocation-', 'master -Blocation- bedroom -Ilocation-', 'dormitory -Blocation- room -Ilocation-',
                'bedchamber -Blocation-', 'chair -Blocation-', 'dryer -Blocation-', 'entrance -Blocation-', 'garden -Blocation-',
                'oven -Blocation-', 'rocking -Blocation- chair -Ilocation-', 'room -Blocation-', 'stove -Blocation-', 'television -Blocation-',
                'washer -Blocation-', 'cellar -Blocation-', 'den -Blocation-', 'laundry -Blocation-', 'pantry -Blocation-', 'patio -Blocation-',
                'balcony -Blocation-', 'lamp -Blocation-', 'window -Blocation-', 'lawn -Blocation-', 'cloakroom -Blocation-', 'telephone -Blocation-',
                'dressing -Blocation- table -Ilocation-', 'bench -Blocation-', 'futon -Blocation-', 'radiator -Blocation-',
                'washing -Blocation- machine -Ilocation-', 'dishwasher -Blocation-']

locations = list(set(locations_at+locations_in+locations_on))


# ================================================================================================================
# names
# ================================================================================================================
names_female = ['hanna -Bperson-', 'barbara -Bperson-', 'samantha -Bperson-', 'erika -Bperson-', 'sophie -Bperson-', 'jackie -Bperson-',
                'skyler -Bperson-', 'jane -Bperson-', 'olivia -Bperson-', 'emily -Bperson-', 'amelia -Bperson-', 'lily -Bperson-',
                'grace -Bperson-', 'ella -Bperson-', 'scarlett -Bperson-', 'isabelle -Bperson-', 'charlotte -Bperson-', 'daisy -Bperson-',
                'sienna -Bperson-', 'chloe -Bperson-', 'alice -Bperson-', 'lucy -Bperson-', 'florence -Bperson-', 'rosie -Bperson-',
                'amelie -Bperson-', 'eleanor -Bperson-', 'emilia -Bperson-', 'amber -Bperson-', 'ivy -Bperson-', 'brooke -Bperson-',
                'summer -Bperson-', 'emma -Bperson-', 'rose -Bperson-', 'martha -Bperson-', 'faith -Bperson-', 'amy -Bperson-',
                'mia -Bperson-', 'sophia -Bperson-', 'abigail -Bperson-', 'isabella -Bperson-', 'ava -Bperson-',
                'katie -Bperson-', 'madison -Bperson-', 'sarah -Bperson-', 'zoe -Bperson-', 'paige -Bperson-']

names_male = ['ken -Bperson-', 'erik -Bperson-', 'samuel -Bperson-', 'skyler -Bperson-', 'brian -Bperson-', 'thomas -Bperson-',
            'edward -Bperson-', 'michael -Bperson-', 'charlie -Bperson-', 'alex -Bperson-', 'john -Bperson-', 'james -Bperson-',
            'oscar -Bperson-', 'peter -Bperson-', 'oliver -Bperson-', 'jack -Bperson-', 'harry -Bperson-', 'henry -Bperson-',
            'jacob -Bperson-', 'thomas -Bperson-', 'william -Bperson-', 'will -Bperson-', 'joshua -Bperson-', 'josh -Bperson-',
            'noah -Bperson-', 'ethan -Bperson-', 'joseph -Bperson-', 'samuel -Bperson-', 'daniel -Bperson-', 'max -Bperson-',
            'logan -Bperson-', 'isaac -Bperson-', 'dylan -Bperson-', 'freddie -Bperson-', 'tyler -Bperson-', 'harrison -Bperson-',
            'adam -Bperson-', 'theo -Bperson-', 'arthur -Bperson-', 'toby -Bperson-', 'luke -Bperson-', 'lewis -Bperson-',
            'matthew -Bperson-', 'harvey -Bperson-', 'ryan -Bperson-', 'tommy -Bperson-', 'michael -Bperson-', 'nathan -Bperson-',
            'blake -Bperson-', 'charles -Bperson-', 'connor -Bperson-', 'jamie -Bperson-', 'elliot -Bperson-', 'louis -Bperson-',
            'liam -Bperson-', 'mason -Bperson-', 'alexander -Bperson-', 'madison -Bperson-',
            'aaron -Bperson-', 'evan -Bperson-', 'seth -Bperson-']

names = list(set(names_male+names_female))
del names_male, names_female

# ================================================================================================================
# what to tell
# ================================================================================================================
what_to_tell_about = ['name -Bwhat_to_tell-', 'nationality -Bwhat_to_tell-', 'eye -Bwhat_to_tell- color -Iwhat_to_tell-',
                    'hair -Bwhat_to_tell- color -Iwhat_to_tell-','surname -Bwhat_to_tell-', 'middle -Bwhat_to_tell- name -Iwhat_to_tell-', 'gender -Bwhat_to_tell-', 'pose -Bwhat_to_tell-',
                    'age -Bwhat_to_tell-', 'job -Bwhat_to_tell-', 'shirt -Bwhat_to_tell- color -Iwhat_to_tell-',
                    'height -Bwhat_to_tell-', 'mood -Bwhat_to_tell-']

what_to_tell_to = [ "your -Bwhat_to_tell- teams -Iwhat_to_tell- affiliation -Iwhat_to_tell-",
                    "your -Bwhat_to_tell- teams -Iwhat_to_tell- name -Iwhat_to_tell-",
                    'the -Bwhat_to_tell- day -Iwhat_to_tell- of -Iwhat_to_tell- the -Iwhat_to_tell- month -Iwhat_to_tell-',
                    'what -Bwhat_to_tell- day -Iwhat_to_tell- is -Iwhat_to_tell- tomorrow -Iwhat_to_tell-',
                    'the -Bwhat_to_tell- time -Iwhat_to_tell-',
                    'the -Bwhat_to_tell- weather -Iwhat_to_tell-',
                    'that -Bwhat_to_tell- i -Iwhat_to_tell- am -Iwhat_to_tell- coming -Iwhat_to_tell-',
                    'to -Bwhat_to_tell- wait -Iwhat_to_tell- a -Iwhat_to_tell- moment -Iwhat_to_tell-',
                    'to -Bwhat_to_tell- come -Iwhat_to_tell- here -Iwhat_to_tell-',
                    'what -Bwhat_to_tell- time -Iwhat_to_tell- is -Iwhat_to_tell- it -Iwhat_to_tell-',
                    'a -Bwhat_to_tell- joke -Iwhat_to_tell-',
                    'something -Bwhat_to_tell- about -Iwhat_to_tell- yourself -Iwhat_to_tell-',
                    'the -Bwhat_to_tell- name -Iwhat_to_tell- of -Iwhat_to_tell- the -Iwhat_to_tell- person -Iwhat_to_tell-']

# ================================================================================================================
# introductions
# ================================================================================================================
intros = ['robot', 'please', 'could you please', 'robot please', 'robot could you please', 'can you', 'robot can you',  'could you', 'robot could you', 'hi', 'hello robot', 'can you please', 'will you', 'will you please', 'kindly']

# Prining number of objects used in the generator
# print('obect_a', len(objects_a))
# print('obect_an', len(objects_an))
# print('objects_the', len(objects_the))
# print('objects_some', len(objects_some))
# print('objects_a_piece_of', len(objects_a_piece_of))
# print('objects_a_cup_of', len(objects_a_cup_of))
# print('objects_a_can_of', len(objects_a_can_of))
# print('objects_a_glass_of', len(objects_a_glass_of))
# print('objects_a_bottle_of', len(objects_a_bottle_of))
print('locations', len(sorted(locations, key=str.lower)))
print('names', len(names))
print('what_to_tell_to', len(what_to_tell_to))
print('intros', len(intros))
print('--------------------------------------