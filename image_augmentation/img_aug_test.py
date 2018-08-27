import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
#=============================================================
class AnnotationPointSet:
    def __init__(self, ann_loc, img_loc, AnimalPoints, ArenaPoints):
        self.AnimalPoints = AnimalPoints
        self.ArenaPoints = ArenaPoints
        self.img_loc = img_loc
        self.ann_loc = ann_loc

class AnimalPointSet:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        
class ArenaPointSet:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
#============================================================
base_path = "C:/Users/Patrick/Desktop/darkflow/flies"

annotation_path = base_path+"/annotations/"
annotations = os.listdir(annotation_path)

annotation_sets = []

for idx, annotation in enumerate(annotations):

	tree = ET.parse(annotation_path+annotation)
	root = tree.getroot()

	arena_sets = []
	animal_sets = []

	img_loc = root.find('path').text
	ann_loc = annotation_path + annotation

	for obj in root.findall('object'):
		if obj.find('name').text == 'arena':

			bb = obj.find('bndbox')

			x_min = int(bb.find('xmin').text)
			x_max = int(bb.find('xmax').text)
			y_min = int(bb.find('ymin').text)
			y_max = int(bb.find('ymax').text)

			new_arena_set = ArenaPointSet(x_min, y_min, x_max, y_max)
			arena_sets.append(new_arena_set)

		else:

			bb = obj.find('bndbox')

			x_min = int(bb.find('xmin').text)
			x_max = int(bb.find('xmax').text)
			y_min = int(bb.find('ymin').text)
			y_max = int(bb.find('ymax').text)

			new_animal_set = AnimalPointSet(x_min, y_min, x_max, y_max)
			animal_sets.append(new_animal_set)

	annotation_sets.append(AnnotationPointSet(ann_loc, img_loc, animal_sets, arena_sets))
#==================================================================
for idx, annotation in enumerate(annotation_sets):

	img_swap_path = annotation.img_loc.replace('\\', '/')
	image = cv2.imread(img_swap_path, 1)

	#=============================================================

	arena_bb_list = []
	for i in range(len(annotation.ArenaPoints)):

		x_min = annotation.ArenaPoints[i].x_min
		x_max = annotation.ArenaPoints[i].x_max
		y_min = annotation.ArenaPoints[i].y_min
		y_max = annotation.ArenaPoints[i].y_max

		arena_bb_list.append(ia.BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max))

	arena_bbs = ia.BoundingBoxesOnImage(arena_bb_list, shape=image.shape)

	#=============================================================

	animal_bb_list = []
	for i in range(len(annotation.AnimalPoints)):

		x_min = annotation.AnimalPoints[i].x_min
		x_max = annotation.AnimalPoints[i].x_max
		y_min = annotation.AnimalPoints[i].y_min
		y_max = annotation.AnimalPoints[i].y_max

		animal_bb_list.append(ia.BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max))

	animal_bbs = ia.BoundingBoxesOnImage(animal_bb_list, shape=image.shape)

	#=============================================================

	seq = iaa.Sequential([
	    iaa.Fliplr(0.75), # horizontal flips
	    iaa.Crop(percent=(0, 0.1)), # random crops
	    iaa.Sometimes(0.5,
	        iaa.GaussianBlur(sigma=(0, 0.5))
	    ),
	    iaa.ContrastNormalization((0.75, 1.5)),
	    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
	    iaa.Multiply((0.8, 1.2), per_channel=0.2),
	    iaa.Affine(
	        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
	        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
	        rotate=(-15, 15),
	        shear=(-3, 3)
	    )
	], random_order=True)

	seq_det = seq.to_deterministic()

	image_aug = seq_det.augment_images([image])[0]
	arena_bbs_aug = seq_det.augment_bounding_boxes([arena_bbs])[0]
	animal_bbs_aug = seq_det.augment_bounding_boxes([animal_bbs])[0]

	#=============================================================

	m = image_aug.copy()

	for i in range(len(arena_bbs.bounding_boxes)):
	    arena_after = arena_bbs_aug.bounding_boxes[i]
	    newImage = cv2.rectangle(m, (int(arena_after.x2), int(arena_after.y2)), (int(arena_after.x1), int(arena_after.y1)), (255,0,0), 3)

	for i in range(len(animal_bbs.bounding_boxes)):
	    animal_after = animal_bbs_aug.bounding_boxes[i]
	    newImage = cv2.rectangle(m, (int(animal_after.x2), int(animal_after.y2)), (int(animal_after.x1), int(animal_after.y1)), (255,0,0), 3)

	#=============================================================

	tree = ET.parse(annotation.ann_loc)
	root = tree.getroot()

	for obj in root.findall('object'):

		arenas = [x for x in obj.findall('name') if x.text=="arena"]
		animals = [x for x in obj.findall('name') if x.text=='animal']

		#=============================================================

		for arena in arenas:
			num = len(arenas)
			for i in range(num):

				bb = obj.find('bndbox')

				bb.find('xmin').text = str(int(animal_after.x1))
				bb.find('xmax').text = str(int(animal_after.x2))
				bb.find('ymin').text = str(int(animal_after.y1))
				bb.find('ymax').text = str(int(animal_after.y2))
		#=============================================================
					
		for animal in animals:
			num = len(animals)
			for i in range(num):

				bb = obj.find('bndbox')

				bb.find('xmin').text = str(int(arena_after.x1))
				bb.find('xmax').text = str(int(arena_after.x2))
				bb.find('ymin').text = str(int(arena_after.y1))
				bb.find('ymax').text = str(int(arena_after.y2))

	#=============================================================
	ann_write_path = 'C:/Users/Patrick/Desktop/darkflow/flies/annotations/aug_img_%s.xml' % idx
	img_write_path = 'C:/Users/Patrick/Desktop/darkflow/flies/images/aug_img_%s.jpg' % idx

	root.find('folder').text = 'images'
	root.find('filename').text = 'aug_img_%s.jpg' % (idx)
	root.find('path').text = img_write_path

	tree.write(ann_write_path)
	plt.imsave(img_write_path, image_aug)
# 	#=============================================================
	    
	test_img_write_path = 'C:/Users/Patrick/Desktop/darkflow/flies/images/tests/%s' % idx
	plt.imsave(test_img_write_path, newImage)