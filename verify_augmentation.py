import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt

def gen_boxes(path_to_annotations):
    
    ann_path = path_to_annotations

    for idx, ann in enumerate(os.listdir(ann_path)):
        tree = ET.parse(path_to_annotations + ann)
        root = tree.getroot()
        img_path = root.find('path').text        
        
        img = cv2.imread(img_path)
        
        box_coords = []
        
        for obj in root.findall('object'):
            bb = obj.find('bndbox')
            
            btm_x = int(bb.find('xmin').text)
            btm_y = int(bb.find('ymin').text)
            top_x = int(bb.find('xmax').text)
            top_y = int(bb.find('ymax').text)
            
            box_coords.append([top_x, top_y, btm_x, btm_y])

            for coordset in box_coords:
                newImage = cv2.rectangle(img, (coordset[0], coordset[1]), (coordset[2], coordset[3]), (255,0,0), 3)
            
        test_img_write_path = 'C:/Users/Patrick/Desktop/darkflow/aug_tests/%s' % idx
        plt.imsave(test_img_write_path, newImage)
    
            
    return None

gen_boxes("C:/Users/Patrick/Desktop/darkflow/flies/annotations/")
 