"""
File: main.py

Author: Vikas Nataraja
Email: viha4393@colorado.edu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os,sys
import random
import argparse
from sklearn.model_selection import train_test_split
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
from keras.models import load_model
from testing import test_model
from train import generate_data, build_model, train_model
from keras import backend as K
K.clear_session()
#tf.reset_default_graph()
#tf.logging.set_verbosity(tf.logging.ERROR)

# Read in arguments from command line
parser = argparse.ArgumentParser()
   
parser.add_argument('--input_csv', default='./train.csv', type=str, 
                    help="Path to input csv file")
parser.add_argument('--train_dir', default='./train_images', type=str, 
                    help="Path to training images directory")
parser.add_argument('--test_dir', default='./test_images', type=str, 
                    help="Path to testing images directory")
parser.add_argument('--model_dir', default='./', type=str, 
                    help="Directory where model will be saved")
parser.add_argument('--model_name', default='weights.h5', type=str, 
                    help="File Name of .h5 file which will contain the weights and the model")
parser.add_argument('--resize_dims', default=(299,299,3), type=tuple, 
                    help="Tuple for new dimensions width x height x channels")
parser.add_argument('--batch_size', default=64, type=int, 
                    help="Batch size for the model")
parser.add_argument('--output_csv', default='./predicted_labels.csv', type=str, 
                    help="Name of csv file to which predicted labels will be saved")

args = parser.parse_args()

# Read in the csv file
train_df = pd.read_csv(args.input_csv)
train_df = train_df[~train_df['EncodedPixels'].isnull()]
train_df['image_name'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
train_df['labels'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
classes = train_df['labels'].unique()
train_df = train_df.groupby('image_name')['labels'].agg(set).reset_index()
for class_name in classes:
    train_df[class_name] = train_df['labels'].map(lambda x: 1 if class_name in x else 0)

# Create a pandas dataframe with one-hot encoded labels
labels_df = {img:vec for img, vec in zip(train_df['image_name'], train_df.iloc[:, 2:].values)}
stratify_split = train_df['labels'].map(lambda x: str(sorted(list(x))))

# Hyperparameters
BATCH_SIZE = args.batch_size
RESIZED_DIMS = args.resize_dims
#print('resize dims=',RESIZED_DIMS)
random_state= 43
test_size = 0.25

# Data augmentation
albumentations_train = Compose([VerticalFlip(), HorizontalFlip(), Rotate(limit=10), GridDistortion()], p=1)

# Split the dataset into training and validation images
train_imgs, val_imgs = train_test_split(train_df['image_name'].values, 
                                        test_size=test_size, 
                                        stratify=stratify_split,
                                        random_state=random_state)

# Create the generator objects for training and validation sets
train_data, val_data = generate_data(train_imgs, val_imgs, labels_df, 
                                     albumentations_train, 
                                     batch_size=BATCH_SIZE,
                                     train_dir=args.train_dir,
                                     resized_dims=RESIZED_DIMS)

# Compile a Keras model
model = build_model(resized_dims=RESIZED_DIMS,train_status=True)

filepath = os.path.join(args.model_dir,args.model_name)

# Train the model on the training and validation images
train_model(model, filepath,
            data_generator_train=train_data,
            data_generator_val=val_data,
            epochs=25,
            steps_per_epoch=60)

# Use the trained model to make predictions
y_predictions = test_model(path_to_model=filepath, batch_size=BATCH_SIZE, test_dir=args.test_dir)

class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
image_names = []
labels = []
for i, (image_name, predictions) in enumerate(zip(os.listdir(args.test_dir), y_predictions)):
    for class_i, class_name in enumerate(class_names):
        image_names.append(image_name)
        labels.append(class_name)

# Save labels predicted as a csv file
pd.DataFrame(list(zip(image_names,predicted_labels)),columns=['image_name','label']).to_csv(args.output_csv)

print('Saved predicted labels as {}'.format(args.output_csv))
