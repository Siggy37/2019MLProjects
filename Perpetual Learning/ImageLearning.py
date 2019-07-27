# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:54:56 2019

@author: brand
"""

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math


class perpetual_learning:
    def __init__(self,k=1, x=2, y=2, seed=200, dimension=65536, centroids=list()):
        self.k = k
        self.x = x
        self.y = y
        self.seed = seed
        self.dimension = dimension
        self.centroids = centroids
        
    def normalize_database(self):
        size = 128, 128
        for i in range(1, 2,1):
            img = Image.open('flower'+str(i)+'.jpg')
#            img.show()
            img.thumbnail(size, Image.ANTIALIAS)
#            img.show()
            img.save('newflower' + str(i) +'.jpg')

        
        img = Image.open('house1.jpg')
        img.thumbnail(size, Image.ANTIALIAS)
        img.save('house1new.jpg')
        """
        img = Image.open('testflower.jpg')
        img.thumbnail(size, Image.ANTIALIAS)
        img.save('testflowernew.jpg')    

        img = Image.open('castle.png')
        img.thumbnail(size, Image.ANTIALIAS)
        img.save('castlenew.png')
        
        img = Image.open('graphicscard.jpg')
        img.thumbnail(size, Image.ANTIALIAS)
        img.save('graphicscardnew.jpg')        
        """
    
    def read_image(self, file):         
        """
        Reads an image file and converts it to a flat vector
        """
        img = Image.open(file).convert('RGBA')
        arr = np.array(img)
        no_noise = []
        for i in range(len(arr)):
            blur = cv2.GaussianBlur(arr[i], (5, 5), 0)
            no_noise.append(blur)
        
        image = np.array(no_noise)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#        Image.fromarray(thresh).show()
        flat_arr = arr.ravel()
        vector = np.matrix(flat_arr)
        vector = vector.tolist()[0]
        max_value = max(vector)
        vector_normalized = [x/max_value for x in vector]
        return np.array(vector_normalized)
        
        """
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        Image.fromarray(sure_fg).show()
        """
    
    
    def generate_starting_point(self, dimension_size):
        np.random.seed(self.seed)
        point = list()
        for i in range(dimension_size):
            feature = np.random.random()
            point.append(feature)
        return np.array(point)
    
    def compute_euclidian_distance(self, centroid, image_vector):
        distance = 0
        for i in range(len(image_vector)):
            partial_distance = ((centroid[i] - image_vector[i]) ** 2)
            distance += partial_distance
        return math.sqrt(distance)
    
    def create_first_centroid(self, image_vectors):
        super_vector = np.zeros(self.dimension)
        for vector in image_vectors:
            super_vector = np.add(super_vector, vector)
        divisor = len(image_vectors) 
        super_vector = [x/divisor for x in super_vector]
        self.centroids.append(super_vector)
        return np.array(super_vector)
            
        
    

if __name__ == '__main__':
    pl = perpetual_learning()
    pl.normalize_database()      

    image_vecs = list()
    for i in range(1, 9,1):
        image_vecs.append(pl.read_image('newflower' + str(i) + '.jpg'))
    centroid0 = pl.create_first_centroid(image_vecs)
    print(centroid0)
    origin_space = list()
    for i in range(1, 9,1):    
        origin_space.append((pl.compute_euclidian_distance(centroid0, pl.read_image('newflower' + str(i) + '.jpg'))))
    m2 = pl.read_image('Mewtwonew.png')
#    print(m2.shape)
    dist = pl.compute_euclidian_distance(centroid0, m2)
    print(max(origin_space))
    if dist > 1.5 * max(origin_space):
        pl.centroids.append(m2)
        
    print(np.array(pl.centroids))
        

