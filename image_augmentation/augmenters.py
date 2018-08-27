import xml.etree.ElementTree as ET
import numpy as np
from cv2 import blur

import random
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_noise(image_array, bbs):
    return [sk.util.random_noise(image_array), bbs]

def horizontal_flip(image_array, bbs):
    
    image_loc = bbs.img_loc
    
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
        
        new = AnnotationPointSet(animal_sets, arena_sets, image_loc)
        
    return [image_array[:, ::-1], new]


def vertical_flip(image_array, bbs):
    
    image_loc = bbs.img_loc
    
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
        
        new = AnnotationPointSet(animal_sets, arena_sets, image_loc)
        
    return [image_array[::-1, :], new]


def blur(image_array, bbs):
    return [blur(image_array, (5, 5)), bbs]

available_transformations = {
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'vertical_flip' : vertical_flip,
    'blur': blur
}