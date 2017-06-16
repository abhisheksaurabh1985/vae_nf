import os
import collections
import tensorflow as tf

# Config data preprocessing
dataset_name= 'mnist_from_tf'
z_score_normalization= True # Not being used right now. 
# Config parameters
z_dim= 2 # Number of dimensions in latent space
h_dim= 256 # Number of latent units
batch_size = 5 # Mini batch size =10
learning_rate = 0.00001 # Learning rate= 0.0001
nepochs= 5
# max_iters = 5000
transfer_fct= tf.nn.softplus # tf.nn.softplus: log(exp(features) + 1)
display_step= 1 # Display logs after these many epochs.
# Config output directories
delete_latent_space_dir= False
delete_data_manifold_dir= False
# Config normalizing flow
normalizing_flow = False
nFlows= 8
flow_type= 'Planar' # 'Planar' or 'Radial'
radial_flow_type= 'Given in the paper on NF' # This is type 1.
invert_condition = True
beta = True

config = collections.OrderedDict(dataset_name = dataset_name,
              z_score_normalization = z_score_normalization,
              z_dim = z_dim,
              h_dim = h_dim,
              batch_size = batch_size,
              learning_rate = learning_rate,
              nepochs = nepochs,
              transfer_fct = transfer_fct,
              display_step = display_step,
              delete_latent_space_dir = delete_latent_space_dir,
              delete_data_manifold_dir = delete_data_manifold_dir,
              normalizing_flow = normalizing_flow,
              nFlows = nFlows,
              flow_type = flow_type,
              radial_flow_type = radial_flow_type,
              invert_condition = invert_condition,
              beta = beta)
        
"""
Hard-coded output directories. To be automated.
"""
if normalizing_flow:
    if flow_type == 'Planar':
        dir_latent_space = './output/norm_flow/mnist_from_tf/planar/latent_space/'
        dir_reconstructed_image = './output/norm_flow/mnist_from_tf/planar/recons/'
        dir_logfile = './output/norm_flow/mnist_from_tf/planar/'
        logfile = os.path.join(dir_logfile, 'logfile.log')
    elif flow_type == 'Radial':
        dir_latent_space = './output/norm_flow/mnist_from_tf/radial/latent_space/'
        dir_reconstructed_image = './output/norm_flow/mnist_from_tf/radial/recons/'
        dir_logfile = './output/norm_flow/mnist_from_tf/radial/'
        logfile = os.path.join(dir_logfile, 'logfile.log')
else:
    dir_latent_space = './output/vae/mnist_from_tf/latent_space/'
    dir_reconstructed_image = './output/vae/mnist_from_tf/recons/'
    dir_logfile = './output/vae/mnist_from_tf/'
    logfile = os.path.join(dir_logfile, 'logfile.log')