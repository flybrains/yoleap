import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2

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
#=============================================================
def random_noise(image_array, bbs):
    return sk.util.random_noise(image_array), bbs

def horizontal_flip(image_array, bbs):
    
    image_loc = bbs.img_loc
    ann_loc = bbs.ann_loc
    
    animal_sets = []
    for i in range(len(bbs.AnimalPoints)):
        x_min = image_array.shape[1] - int(bbs.AnimalPoints[i].x_max)
        x_max = image_array.shape[1] - int(bbs.AnimalPoints[i].x_min)
        
        new_animal_set = AnimalPointSet(x_min, bbs.AnimalPoints[i].y_min, x_max, bbs.AnimalPoints[i].y_max)
        animal_sets.append(new_animal_set)

    arena_sets = []
    for i in range(len(bbs.ArenaPoints)):
        x_min = image_array.shape[1] - int(bbs.ArenaPoints[i].x_max)
        x_max = image_array.shape[1] - int(bbs.ArenaPoints[i].x_min)
        
        new_arena_set = ArenaPointSet(x_min, bbs.ArenaPoints[i].y_min, x_max, bbs.ArenaPoints[i].y_max)
        arena_sets.append(new_arena_set)
        
        new = AnnotationPointSet(ann_loc, image_loc, animal_sets, arena_sets)
        
    return image_array[:, ::-1], new


def vertical_flip(image_array, bbs):
    
    image_loc = bbs.img_loc
    ann_loc = bbs.ann_loc
    
    animal_sets = []
    for i in range(len(bbs.AnimalPoints)):
        y_min = image_array.shape[0] - int(bbs.AnimalPoints[i].y_max)
        y_max = image_array.shape[0] - int(bbs.AnimalPoints[i].y_min)
        
        new_animal_set = AnimalPointSet(bbs.AnimalPoints[i].x_min, y_min, bbs.AnimalPoints[i].x_max, y_max)
        animal_sets.append(new_animal_set)

    arena_sets = []
    for i in range(len(bbs.ArenaPoints)):
        y_min = image_array.shape[0] - int(bbs.ArenaPoints[i].y_max)
        y_max = image_array.shape[0] - int(bbs.ArenaPoints[i].y_min)
        
        new_arena_set = ArenaPointSet(bbs.ArenaPoints[i].x_min, y_min, bbs.ArenaPoints[i].x_max, y_max)
        arena_sets.append(new_arena_set)
        
        new = AnnotationPointSet(ann_loc, image_loc, animal_sets, arena_sets)
        
    return image_array[::-1, :], new


def blur(image_array, bbs):
    return blur(image_array, (5, 5)), bbs

#=============================================================

base_path = "C:/Users/Patrick/Desktop/darkflow/flies"

#=============================================================

image_path = base_path+"/images/"

annotation_path = base_path+"/annotations/"
annotations = os.listdir(annotation_path)

#=============================================================

annotation_sets = []

for annotation in annotations:

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

# #=============================================================

num_files_per_frame = 2
num_generated_files = 0

for idx, annotation in enumerate(annotation_sets):

	img_swap_path = annotation.img_loc.replace('\\', '/')

	image = cv2.imread(img_swap_path, -1)

	# Iterate through annotation objects

	for i in range(num_files_per_frame):

		num_trans_to_apply = random.randint(1, 4)

		num_trans_applied = 0

	 	# Apply random transformations to image and annotation object

		while num_trans_applied <= num_trans_to_apply:
				
			index = random.randint(1,4)

			# if index == 1:
			# 	image, annotation = blur(image, annotation)

			if index == 2:
				image, annotation = random_noise(image, annotation)

			if index == 3:
					image, annotation = horizontal_flip(image, annotation)

			else:
				image, annotation = vertical_flip(image, annotation)

			num_trans_applied += 1

		new_annotation = annotation

		# Create new path to save image at
		new_img_path = '%s/images/aug_img_%s.jpg' % (base_path, num_generated_files)
		plt.imsave(new_img_path, image)

		# Create new XML tree to match augmented image
		# Open current annotation for ground truth image
		new_tree = ET.parse(annotation.ann_loc)

		#Immediately copy to new file
		new_annotation_path = '%s/annotations/aug_img_%s.xml' % (base_path, num_generated_files)
		new_tree.write(new_annotation_path)


		#Open new copy
		copy_tree = ET.parse(new_annotation_path)
		root = copy_tree.getroot()

		# Parse through objects and add to proper elements of annotation object
		for obj in root.findall('object'):

			arenas = [x for x in obj.findall('name') if x.text=="arena"]
			animals = [x for x in obj.findall('name') if x.text=='animal']

			for arena in arenas:
				num = len(arenas)
				for i in range(num):

					bb = obj.find('bndbox')

					bb.find('xmin').text = str(new_annotation.ArenaPoints[i].x_min)
					bb.find('xmax').text = str(new_annotation.ArenaPoints[i].x_max)
					bb.find('ymin').text = str(new_annotation.ArenaPoints[i].x_min)
					bb.find('ymax').text = str(new_annotation.ArenaPoints[i].x_max)

			for animal in animals:
				num = len(animals)
				for i in range(num):

					bb = obj.find('bndbox')

					bb.find('xmin').text = str(new_annotation.AnimalPoints[i].x_min)
					bb.find('xmax').text = str(new_annotation.AnimalPoints[i].x_max)
					bb.find('ymin').text = str(new_annotation.AnimalPoints[i].y_min)
					bb.find('ymax').text = str(new_annotation.AnimalPoints[i].y_max)


		root.find('folder').text = 'images'
		root.find('filename').text = 'aug_img_%s.jpg' % (num_generated_files)
		root.find('path').text = '%s/images/aug_img_%s.jpg' % (base_path, num_generated_files)

		copy_tree.write('%s/annotations/aug_img_%s.xml' % (base_path, num_generated_files))

		num_generated_files += 1