"""
File: testing.py

Author: Vikas Nataraja
Email: viha4393@colorado.edu
"""

import numpy as np
import keras
import os
from DataGenerator import DataGenerator
from keras.models import load_model


def test_model(path_to_model, test_dir, batch_size, resized_dims=(299,299,3)):
    """
    Args:
        - path_to_model: path to the model (h5 file) to be loaded
        - test_dir: directory containing the test images
        - batch_size: batch size for the generator
        - resized_dims: dimensions to which image will be resized to.
    Returns:
        - y_pred: numpy array containing label probabilities
    """
    model = load_model(path_to_model)
    data_generator_test = DataGenerator(dir_imgs=test_dir,
                                        shuffle=False, 
                                        batch_size=batch_size,
                                        resized_dims=resized_dims)
    
    y_pred = model.predict_generator(data_generator_test, verbose=1)
    print('Finished predictions successfully, exiting testing...\n')
    
    return y_pred




