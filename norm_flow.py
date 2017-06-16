# Import native libraries
import time
import os
import shutil

# Import third party libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To build TF from source. Supposed to speed up the execution by 4-8x. 
import tensorflow as tf
from scipy.stats import norm
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.python import debug as tf_debug

# import matplotlib.gridspec as gridspec

# Import files
from config import *
from data_source import * # Functions related to fetching data
from utilities import *
from distributions import *
from plots import *
from losses import *
from apply_flow import *
from train import *

# Import class files
from NormalizingRadialFlow import NormalizingRadialFlow
from NormalizingPlanarFlow import NormalizingPlanarFlow
from NeuralNetwork import *

tf_version = tf.__version__ # Tensor flow version
print "Tensor Flow version {}".format(tf_version) 

np.random.seed(0)
tf.set_random_seed(0)

# Get the name of this script. __file__ does not work on the shell. 
filename_script = os.path.basename(os.path.realpath(__file__))
print filename_script

np.random.seed(1234) # reproducibility

"""
Load data
"""
if dataset_name == 'real-valued-mnist':
    print "Using real valued MNIST dataset"
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval() # _t denotes the targets.
#    print "Number of training samples {}; Shape of training data[{}]; \
#           Shape of training labels[{}]".format(train_x.shape[0], train_x.shape, train_t.shape)
           
if dataset_name== 'mnist_from_tf':
    from tensorflow.examples.tutorials.mnist import input_data
    print "Using MNIST from Tensorflow"
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # n_samples = mnist.train.num_examples
    train_x, valid_x, test_x= mnist.train.images, mnist.validation.images, mnist.test.images
    train_t, valid_t, test_t= mnist.train.labels, mnist.validation.labels, mnist.test.labels
    # print "Number of samples {}; Shape of y[{}]; Shape of X[{}]".format(n_samples, mnist.train.labels.shape, mnist.train.images.shape)
    # plt.imshow(np.reshape(-mnist.train.images[4242], (28, 28)), interpolation='none',cmap=plt.get_cmap('gray'))
    # mnist.train.images.min() # Pixel values are between 0 and 1. 

# Concatenate training and validation set
train_x = np.concatenate([train_x,valid_x])
n_samples= train_x.shape[0] # Total number of samples including those in validation set.
# train_t = np.concatenate([train_t,valid_t]) # This will need some work. In tf the labels are one-hot encoded.
print "Number of training samples {}; Shape of training data[{}]; \
       Shape of training labels[{}]".format(train_x.shape[0], train_x.shape, train_t.shape)

"""
Set up output directories

THIS PART NEEDS TO BE RE-DONE.
"""
#results_out, logfile, latent_space, data_manifold = \
#                        manage_output_directories(filename_script, 
#                                                  dataset_name, 
#                                                  delete_data_manifold_dir = False,
#                                                  delete_latent_space_dir = False)
    
"""
Write config in logfile
"""
with open(logfile,'a') as fl:
    fl.write("########\n")
    fl.write("Config\n")
    fl.write("########\n")
    for k,v in config.items():
        fl.write('{0}: {1}\n'.format(k, v))

"""
Encoder
"""
X_dim = train_x.shape[1] # Input dimension 
# Placeholders for input and latent space
X, z = inputs(X_dim, z_dim)
nn = NeuralNetwork(X_dim, h_dim, z_dim, transfer_fct = tf.nn.softplus)

if normalizing_flow:
    # z_mu, z_log_var, z0, flow_params = nn.encoder(X, z, X_dim, h_dim, z_dim, nFlows)
    z_mu, z_log_var, flow_params = nn.encoder_nf(X, z, X_dim, h_dim, z_dim, nFlows)
    z_var= tf.exp(z_log_var) # Get variance
else:
    z_mu, z_log_var= nn.enc_vanilla_vae(X) 
    z_var= tf.exp(z_log_var) # Get variance
     

# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nn.sample_z(z_mu, z_var)

"""
Flow
"""
if normalizing_flow:
    if flow_type == "Planar":
        currentClass = NormalizingPlanarFlow(z0, z_dim)
        z_k, sum_log_detj = currentClass.planar_flow_2(z0, flow_params, nFlows, 
                                                       z_dim, invert_condition)
    elif flow_type == "Radial":
        if radial_flow_type == "Given in the paper on NF":
            currentClass = NormalizingRadialFlow(z0, z_dim, radial_flow_type)
            z_k, sum_log_detj = currentClass.radial_flow(z0, flow_params, nFlows, 
                                                           z_dim, invert_condition)
else:
    z_k = z0

"""
Decoder
"""
# out_op: reconstructed image
x_recons, x_recons_mean, x_recons_logvar = nn.decoder(z_k, X_dim, h_dim, z_dim)
# out_op, out_mu, out_log_var = nn.decoder(z_k, X_dim, h_dim, z_dim)

"""
Loss
"""
# loss_op = make_loss(out_op, X, z_log_var, z_mu, log_detj, z0)
if normalizing_flow:
    global_step = tf.Variable(0, trainable=False)
    loss_op = elbo_loss(X, x_recons, beta, global_step, recons_error_func = cross_entropy_loss,
                                              z_mu=z_mu, z_var= z_var, z0= z0, 
                                              zk= z_k, logdet_jacobian= sum_log_detj)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, 
                                                              global_step=global_step)
else:
    """
    Use the elbo_loss function at this step. Modify the following line.
    """
    global_step = tf.Variable(0, trainable=False)
    loss_op = elbo_loss(X, x_recons, beta, global_step, recons_error_func = cross_entropy_loss, 
                                              z_mu=z_mu, z_var= z_var)
    # loss_op = vanilla_vae_loss(X, x_recons, z_mu, z_var)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op,
                                                              global_step= global_step)


sess = tf.InteractiveSession()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

tf.summary.FileWriter("tf_logs", graph=sess.graph)

"""
Train
"""
avg_free_energy_bound = train_norm_flow(n_samples, dir_latent_space, logfile, z_mu, 
                                 z_var, sess, train_op, X, loss_op, mnist)

"""
Plots
"""
# Plot actual and the reconstructed digit side by side. 
plot_reconstruction(X, sess, x_recons_mean, dataset_name)
plt.savefig(os.path.join(dir_reconstructed_image, 'recons.png'))
# Plot of digit manifold with 2D latent space.
plot_latent_space(X, sess, z_mu, mnist, dataset_name = 'mnist_from_tf')
plt.savefig(os.path.join(dir_latent_space, 'latent_space.png'))
#plot_digits_in_latent_space(sess, prob_z_sample, z_sample, mnist, dataset_name,
#                            latent_dim= z_dim)
## Plot loss vs epoch
#plot_loss_vs_epoch(range(nepochs), avg_free_energy_bound)   

sess.close()
