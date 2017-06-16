# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:16:48 2017

@author: abhishek
"""
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import os
import shutil

# Import class files
# from VariationalAutoencoder import VariationalAutoencoder

def inputs(D, Z):
    """
    D: Input dimension
    Z: Latent space dimension
    """
    X = tf.placeholder(tf.float32, shape = [None, D], name = 'X')
    z = tf.placeholder(tf.float32, shape = [None, Z], name = 'z')
    return X, z

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier/ Glorot initialization of network weights
    constant=1 is for tanh activation
    """
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
                             
def manage_output_directories(filename_script, dataset_name, 
                              delete_data_manifold_dir = False,
                              delete_latent_space_dir = False):
    """
    TO BE MODIFIED
    """
    results_out = os.path.join("output", os.path.splitext(filename_script)[0], dataset_name)
    
    if not os.path.exists(results_out):
        os.makedirs(results_out)
    shutil.copy(os.path.join(os.path.realpath(__file__)[0:90], filename_script), os.path.join(results_out, filename_script))
    logfile = os.path.join(results_out, 'logfile.log')
    
    # Folder for storing 2D latent space every epoch. These will be used to make GIFs or a vide. 
    latent_space= os.path.join(results_out, 'latent_space')  
    if not os.path.exists(latent_space):
        os.makedirs(latent_space)
        
    # Folder for storing data manifold. 
    data_manifold= os.path.join(results_out, 'data_manifold')
    if not os.path.exists(data_manifold):
        os.makedirs(data_manifold)

    # Empty the graphs of the latent space generated in the previous run
    if delete_latent_space_dir:
        delete_files_in_directory(latent_space)
    
    # Empty the graphs of the latent space generated in the previous run
    if delete_data_manifold_dir:
        delete_files_in_directory(data_manifold) 
    
    return results_out, logfile, latent_space, data_manifold    

def delete_files_in_directory(dirname):
    '''
    To remove subdirectories too, uncomment the elif statement.
    param dirname: Directory which is to be emptied.
    '''
    for each_file in os.listdir(dirname):
        file_path = os.path.join(dirname, each_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e) 
    return       
    
