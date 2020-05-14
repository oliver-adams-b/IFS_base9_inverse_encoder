from __future__ import absolute_import, division, print_function
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import keras.backend as kb
import os
import pickle
import datetime

kb.set_image_dim_ordering('tf')
kb.set_image_data_format = 'channels last'

#loading the the training and testing data
path = 'base_9_inverse_problem_data/training_data/'
dir_list = os.listdir(path) 

all_files = []

for file_name in dir_list:
    with open(path + file_name, 'rb') as file:
        all_files.append(pickle.load(file))
     
all_files = np.array(all_files)

training_images, training_mats = [], []
testing_images, testing_mats = [], []

l = len(all_files[:, 0])
n = 6500

training_images = np.array([1 - x.reshape(100, 100, 1)/254 for x in all_files[:n,0]])
training_mats =  np.array([x.reshape(9, 9, 1) for x in all_files[:n, 1]])

testing_images = np.array([1 - x.reshape(100, 100, 1)/254 for x in all_files[n:,0]])
testing_mats =  np.array([ x.reshape(9, 9, 1)  for x in all_files[n:, 1]])

def custom_loss(y_true, y_predict): 
    #doesn't return a scalar but we will get back to that maybe
    return 9 - kb.cumsum(kb.eye(9) - kb.dot(y_predict, tf.linalg.inv(y_true)))  

def ssim_loss(y_true, y_predict):   
    return -1*tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_predict, 1.0))

#create model
def get_model():
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(filters = 5, 
                                  kernel_size = (6, 6), 
                                  activation=tf.nn.relu, 
                                  input_shape=(100,100, 1)))
    
    model.add(keras.layers.Conv2D(filters = 4, 
                                  kernel_size = (5, 5), 
                                  activation='relu', 
                                  padding = 'valid', 
                                  use_bias = True))
    
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters = 2, 
                                  kernel_size = (3, 3), 
                                  activation='relu', 
                                  padding = 'valid', 
                                  use_bias = True))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters = 2, 
                                  kernel_size = (2, 2), 
                                  activation='relu', 
                                  padding = 'valid', 
                                  use_bias = True))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters = 1, 
                                  kernel_size = (2, 2), 
                                  activation='relu',
                                  padding = 'valid', 
                                  use_bias = True))
    return model

#using the pretty tensorboard system to display model progress
log_dir="logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d")
tensorboard_callback = TensorBoard(log_dir=log_dir)

model = get_model()
nepochs = 

model.summary()

model.compile(optimizer='adagrad', 
              loss='categorical_hinge', 
              metrics = ['mae'])

model.fit(training_images, 
          training_mats,
          validation_data=(testing_images, testing_mats), 
          epochs=nepochs, callbacks=[tensorboard_callback])

model.save("\models\initial_base9_inverse_model_max_pool.h5")


