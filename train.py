"""
File: train.py

Author: Vikas Nataraja
Email: viha4393@colorado.edu
"""
import pandas as pd
import numpy as np
import keras
import os
import random
from DataGenerator import DataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam
from keras.layers import Dense, GlobalMaxPool2D, Dropout, Flatten
from keras.models import Model

def generate_data(train_imgs, val_imgs, labels_df, albumentations_train, batch_size, train_dir, resized_dims):
    """
    Args:
        - train_imgs: list of images to be used for training
        - val_imgs: list of images to be used for validation
        - albumentations_train: data augmentation from albumentations package
        - batch_size: batch size for the generator
        - train_dir: directory containing training images
        - resized_dims: dimensions to which images will be resized to
    Returns:
        - data_generator_train: generator object for training images
        - data_generator_val: generator object for validation images
    """

    data_generator_train = DataGenerator(train_imgs, label_vector=labels_df, 
                                         dir_imgs=train_dir,
                                         resized_dims=resized_dims,
                                         batch_size=batch_size,
                                         augmentation=albumentations_train,
                                         shuffle=True)

    data_generator_val = DataGenerator(val_imgs, label_vector=labels_df,
                                       dir_imgs=train_dir,
                                       batch_size=batch_size,
                                       resized_dims=resized_dims,
                                       shuffle=False)
    
    return (data_generator_train, data_generator_val)

def build_model(classes=4, learning_rate=0.001,
                resized_dims=(299,299,3),
                dropout_probability=0.5,
                train_status = True,
                loss_function='binary_crossentropy',
                accuracy_function='accuracy'):
    """
    Args:
        - classes: number of classes in output
        - learning_rate: learning rate for optimizer
        - resized_dims: dimensions to which image will be resized to
        - dropout_probability: probability value between 0 and 1 for dropout layer
        - train_status: Boolean, set to True if weights need to be updated during training, set to 
                        False if weights need to be frozen
        - loss_function: loss function for model
        - accuracy_function: accuracy function for model
    Returns:
        - model: Keras model with compiled architectural layers
    """
    
    backbone_model = InceptionV3(include_top=False, weights=None, input_shape=resized_dims)

    CLASSES = classes
    x = backbone_model.output
    #x = GlobalMaxPool2D()(x)
    # Add a dropout layer with dropout probability of 0.5 by default
    x = Dropout(dropout_probability)(x)
    
    # Add a Flatten layer to make dimensions compatible
    x = Flatten()(x)
    
    # Add Dense (Fully-Connected Layer)
    predictions = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=backbone_model.input, outputs=predictions)
    
    # trainable status - whether to freeze weights or not. Setting trainable=True will mean weights are updated
    for layer in backbone_model.layers:
        layer.trainable = train_status

    model.compile(optimizer=Adam(learning_rate=learning_rate,clipnorm=1.,clipvalue=0.5),
                  loss=loss_function,
                  metrics=[accuracy_function])
    
    return model

def train_model(model, filepath, data_generator_train, data_generator_val, epochs=25, steps_per_epoch=50):
    """
    Args:
        - model: Keras model which has already been compiled
        - filepath: path where weights need to be saved
        - data_generator_train: generator object for training images
        - data_generator_val: generator object for validation images
        - epochs: number of epochs for training
        - steps_per_epoch: number of steps per epoch
    Returns:
        void
    """    
    checkpoint = ModelCheckpoint(filepath, save_best_only=True, verbose=1, period=1)
    print('Model will be saved in directory: {} as {}\n'.format(os.path.split(filepath)[0],os.path.split(filepath)[1]))
    model.fit_generator(data_generator_train,
                        validation_data=data_generator_val,
                        callbacks=[checkpoint],
                        epochs=epochs,verbose=1,
                        steps_per_epoch=steps_per_epoch)
    print('Finished training model. Exiting function ...\n')
