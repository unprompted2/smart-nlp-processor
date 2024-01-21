import random
import pickle

objects_a = ['apple -Itheme-', 'bath -Itheme- towel -Itheme-', 'towel -Itheme-', 'book -Itheme-', 'box -Itheme-', 'can -Itheme-', 'cloth -Itheme-',
			'coat -Itheme-', 'coffee -Itheme-', 'coke -Itheme-', 'cover -Itheme-', 'cutlery -Itheme-', 'folder -Itheme-', 'fruit -Itheme-',
			'handbag -Itheme-', 'jacket -Itheme-', 'jam -Itheme-', 'jam -Itheme- jar -Itheme-', 'knife -Itheme-', 'lamp -Itheme-', 'lanyard -Itheme-',
			'laundry -Itheme-', 'magazine -Itheme-', 'milk -Itheme-', 'mug -Itheme-', 'mustard -Itheme-', 'newspaper -Itheme-',
			'paper -Itheme-', 'phone -Itheme-', 'pillow -Itheme-', 'salt -Itheme-', 'screwdriver -Itheme-', 'soap -Itheme-', 't-shirt -Itheme-', 'tablet -Itheme-',
			'toilet paper -Itheme-', 'towel -Itheme-', 'trash -Itheme-', 'tray -Itheme-', 'vase -Itheme-',
			'wallet -Itheme-', 'water -Itheme-', 'wine -Itheme-', 'yogurt -Itheme-']

objects_the = ['apple -Itheme-', 'bath -Itheme- towel -Itheme-', 'towel -Itheme-', 'book -Itheme-', 'box -Itheme-', 'can -Itheme-', 'cloth -Itheme-',
			'coat -Itheme-', 'coffee -Itheme-', 'coke -Itheme-', 'cover -Itheme-', 'cutlery -Itheme-', 'folder -Itheme-', 'fruit -Itheme-',
			'handbag -Itheme-', 'jacket -Itheme-', 'jam -Itheme-', 'jam -Itheme- jar -Itheme-', 'knife -Itheme-', 'lamp -Itheme-', 'lanyard -Itheme-',
			'laundry -Itheme-', 'magazine -Itheme-', 'milk -Itheme-', 'mug -Itheme-', 'mustard -Itheme-', 'newspaper -Itheme-',
			'paper -Itheme-', 'phone -Itheme-', 'pillow -Itheme-', 'salt -Itheme-', 'screwdriver -Itheme-', 'soap -Itheme-', 't-shirt -Itheme-', 'tablet -Itheme-',
			'toilet -Itheme- paper -Itheme-', 'towel -Itheme-', 'trash -Itheme-', 'tray -Itheme-', 'vase -Itheme-',
			'wallet -Itheme-', 'water -Itheme-', 'wine -Itheme-', 'yogurt -Itheme-']

objects_a_cup_of = ['milk -Itheme-', 'coffee -Itheme-', 'water -Itheme-', 'wine -Itheme-']

objects_a_can_of = ['milk -Itheme-', 'coffee -Itheme-', 'water -Itheme-', 'wine -Itheme-']

objects_a_glass_of = ['milk -Itheme-', 'coffee -Itheme-', 'water -Itheme-', 'wine -Itheme-']

objects_a_bottle_of = ['milk -Itheme-', 'coffee -Itheme-', 'water -Itheme-', 'wine -Itheme-']


locations = ['washing -Ilocation- machine -Ilocation-', 'wardrobe -Ilocation-', 'tv -Ilocation-', 'television -Ilocation-', 'table -Ilocation-',
			'kitchen -Ilocation- table -Ilocation-', 'dining -Ilocation- table -Ilocation-', 'studio -Ilocation-', 'stove -Ilocation-', 'sofa -Ilocation-',
			'sink -Ilocation-', 'shower -Ilocation-', 'room -Ilocation-', 'restroom -Ilocation-', 'refrigerator -Ilocation-', 'nightstand -Ilocation-',
			'mirror -Ilocation-', 'loo -Ilocation-', 'living -Ilocation- room -Ilocation-', 'kitchen -Ilocation-', 'dresser -Ilocation-', 'dishwasher -Ilocation-',
			'dining -Ilocation- room -Ilocation-', 'cupboard -Ilocation-', 'counter -Ilocation-', 'couch -Ilocation-', 'closet -Ilocation-', 'bin -Ilocation-',
			'bedroom -Ilocation-', 'bathroom -Ilocation-', 'shower -Ilocation-']

names = ['daniele -Ibeneficiary-', 'john -Ibeneficiary-', 'martina -Ibeneficiary-', 'michael -Ibeneficiary-', 'vittorio -Ibeneficiary-', 'guest -Ibeneficiary-'] 

intros = ['robot', 'mbot', 'monarch', 'please', 'could you please', 'robot please', 'mug you', 'mbot please', 'robot mug you', 'mbot could you',
		'robot could you']

tasks = []
tasks_motion = []
tasks_searching = []
tasks_taking = []
tasks_placing = []
tasks_bringing = []
tasks_other = []

#-----------------------------------------PLACING---------------------------------------------
for objet in objects_a:

	task = 'place a ' + objet

	tasks_placing.append(task)

	task = 'place them -Btheme- '

	tasks_placing.append(task)

	task = 'put it -Btheme- '