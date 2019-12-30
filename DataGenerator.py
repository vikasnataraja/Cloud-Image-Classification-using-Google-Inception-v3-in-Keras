"""
File: DataGenerator.py

Author: Vikas Nataraja
Email: viha4393@colorado.edu
"""

from copy import deepcopy
import numpy as np
import cv2
from keras.utils import Sequence
import os

class DataGenerator(Sequence):
    """
    Class to generate images of a certain batch size
    
    Args:
        - images_list: list of images to be used
        - label_vector: list of corresponding labels
        - dir_imgs: directory containing the images
        - batch_size: batch size for the generator
        - shuffle: boolean, shuffles dataset if True
        - augmentation: data augmentation from albumentations package
        - resized_dims: dimensions to which images will be resized to
    Returns:
        - X: image tensor of the format (batch_size, width, height, channels)
        - y: corresponding labels
    """
    def __init__(self, images_list=None, label_vector=None, dir_imgs='./train_images',
                 batch_size=64, shuffle=True, augmentation=None,
                 resized_dims=(480,480,3)):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        if images_list is None:
            self.images_list = os.listdir(dir_imgs)
        else:
            self.images_list = deepcopy(images_list)
        #print(self.images_list)
        self.dir_imgs = dir_imgs
        self.len = len(self.images_list) // self.batch_size
        self.resized_dims = resized_dims
        self.num_classes = 4
        self.label_vector = label_vector
        #self.mode = mode
        #self.is_test = not 'train' in dir_imgs
        #if 'train' in self.mode:
        #self.labels = [self.label_vector[img] for img in self.images_list[:self.len*self.batch_size]]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        current_batch = self.images_list[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.empty((self.batch_size, self.resized_dims[0], self.resized_dims[1], self.resized_dims[2]))
        y = np.empty((self.batch_size, self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.dir_imgs, image_name)
            img = cv2.resize(cv2.imread(path), (self.resized_dims[0], self.resized_dims[1])).astype(np.float32)
            img = (img - np.min(img))/(np.max(img)-np.min(img))
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img
            #if 'train' in self.mode:
            #self.labels = [self.label_vector[img] for img in self.images_list[:self.len*self.batch_size]]
            if self.label_vector is not None:
                y[i, :] = self.label_vector[image_name]
        return X, y
    