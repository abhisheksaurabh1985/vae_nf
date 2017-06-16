# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:49:08 2017

@author: abhishek
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import utilities
from config import *
from utilities import *

class NeuralNetwork(object):

    def __init__(self, X_dim, h_dim, z_dim, transfer_fct = tf.nn.softplus):
        
        self.X_dim = X_dim # Number of input neurons
        self.h_dim = h_dim # Number of neurons in the hidden layer. Same for both encoder and decoder.
        self.z_dim = z_dim # Number of dimensions in latent space.
        self.transfer_fct = transfer_fct # Transfer function. Same for both encoder and decoder.

        # self.X = tf.placeholder(tf.float32, shape=[None, X_dim], name="X")
        # self.z = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")
        
        # Weight initialization for encoder    
        self.Q_W1 = tf.Variable(xavier_init(self.X_dim, self.h_dim))
        self.Q_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.Q_W2_mu = tf.Variable(xavier_init(self.h_dim, self.z_dim))
        self.Q_b2_mu = tf.Variable(tf.zeros(shape=[self.z_dim]))
        self.Q_W2_sigma = tf.Variable(xavier_init(self.h_dim, self.z_dim))
        self.Q_b2_sigma = tf.Variable(tf.zeros(shape=[self.z_dim]))
        
        # Weight for normalizing flow parameters. 
        self.Q_w_us = tf.Variable(xavier_init(self.h_dim, nFlows * self.z_dim))
        self.Q_b_us = tf.Variable(tf.zeros(shape=[nFlows * self.z_dim]))
        self.Q_w_ws = tf.Variable(xavier_init(self.h_dim, nFlows * self.z_dim))
        self.Q_b_ws = tf.Variable(tf.zeros(shape=[nFlows * self.z_dim]))
        self.Q_w_bs = tf.Variable(xavier_init(self.h_dim, nFlows))  
        self.Q_b_bs = tf.Variable(tf.zeros(shape=[nFlows]))
        
        # Weight initialization for decoder        
        self.P_W1 = tf.Variable(xavier_init(self.z_dim, self.h_dim))
        self.P_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.P_W2 = tf.Variable(xavier_init(self.h_dim, self.X_dim))
        self.P_b2 = tf.Variable(tf.zeros(shape=[self.X_dim]))
        
        # init = tf.global_variables_initializer()
        init = tf.global_variables_initializer() # Initializing the tensor flow variables
        self.sess = tf.InteractiveSession()  # Launch the session
        self.sess.run(init)
    
    def enc_vanilla_vae(self, X):
        '''
        This is the recognition network (probabilistic encoder). Maps inputs onto
        a normal distribution in latent space. Returns mean and log variance of the 
        Gaussian distribution in the latent space. 
        params X: input data as a tensor
        Returns:
        z_mu: mean of the Gaussian distribution in the latent space
        z_logvar: log variance of the Gaussian distribution in the latent space 
        '''
        h = self.transfer_fct(tf.matmul(X, self.Q_W1) + self.Q_b1) # Try with tf.nn.softplus or tf.tanh.
        self.z_mu = tf.matmul(h, self.Q_W2_mu) + self.Q_b2_mu
        self.z_logvar = tf.matmul(h, self.Q_W2_sigma) + self.Q_b2_sigma
        
#        # Normalizing Flow parameters
#        self.us = tf.matmul(h, self.Q_w_us) + self.Q_b_us
#        self.ws = tf.matmul(h, self.Q_w_ws) + self.Q_b_ws
#        self.bs = tf.matmul(h, self.Q_w_bs) + self.Q_b_bs
#
#        flow_params = (self.us, self.ws, self.bs)
        
        return self.z_mu, self.z_logvar #, flow_params
    
    def sample_z(self, z_mu, z_var):
        '''
        Reparametrization trick. Transforms the standard normal to a distribution 
        with mean and sigma outputted by the encoder. 
        param mu: 
        param log_var:
        
        '''
        # eps = tf.random_normal(shape=tf.shape(z_mu), mean= 0, stddev= 1)
        eps = tf.random_normal(shape=tf.shape(z_mu), mean= 0, stddev= 1)
        _z = tf.add(z_mu, tf.multiply(tf.sqrt(z_var), eps))
        return _z # Elementwise product between the variance and noise. If mean and logvar are (minibatchsize*2), noise is also of the same size.    

    def P(self, z):
        """
        NOT BEING USED.
        """
        h = self.transfer_fct(tf.matmul(z, self.P_W1) + self.P_b1)
        # logits = tf.matmul(h, P_W2) + P_b2
        # prob = tf.nn.sigmoid(logits) # Computer sigmoid of x element wise. Returns a tensor with the same type as input. 
        prob = tf.nn.sigmoid(tf.matmul(h, self.P_W2) + self.P_b2) # Computer sigmoid of x element wise. Returns a tensor with the same type as input. 
        return prob      
        
    def encoder_nf(self, x, e, D, H= h_dim, Z= z_dim, K= nFlows, initializer=tf.contrib.layers.xavier_initializer):
        """
        X: Input tf placeholder
        z: Latent space tf placeholder
        D: Data dimension
        H: Number of hidden neurons
        Z: Latent Dimension
        K: Number of flow
        """
        with tf.variable_scope('encoder', reuse = None):
            w_h = tf.get_variable('w_h', [D, H], initializer=initializer())
            b_h = tf.get_variable('b_h', [H])
            w_mu = tf.get_variable('w_mu', [H, Z], initializer=initializer())
            b_mu = tf.get_variable('b_mu', [Z])
            w_v = tf.get_variable('w_v', [H, Z], initializer=initializer())
            b_v = tf.get_variable('b_v', [Z])
    
            # Weights for outputting normalizing flow parameters
            w_us = tf.get_variable('w_us', [H, K*Z])
            b_us = tf.get_variable('b_us', [K*Z])
            w_ws = tf.get_variable('w_ws', [H, K*Z])
            b_ws = tf.get_variable('b_ws', [K*Z])
            w_bs = tf.get_variable('w_bs', [H, K])
            b_bs = tf.get_variable('b_bs', [K])
    
            h = tf.nn.relu(tf.matmul(x, w_h) + b_h)
            mu = tf.matmul(h, w_mu) + b_mu
            log_var = tf.matmul(h, w_v) + b_v
            # z = mu + tf.sqrt(tf.exp(log_var))*e
    
            # Normalizing Flow parameters
            us = tf.matmul(h, w_us) + b_us
            ws = tf.matmul(h, w_ws) + b_ws
            bs = tf.matmul(h, w_bs) + b_bs
    
            lambd = (us, ws, bs)
    
        # return mu, log_var, z, lambd
        return mu, log_var, lambd

    def encoder_nf_radial(self, x, e, D, H= h_dim, Z= z_dim, K= nFlows, initializer=tf.contrib.layers.xavier_initializer):
        """
        X: Input tf placeholder
        z: Latent space tf placeholder
        D: Data dimension
        H: Number of hidden neurons
        Z: Latent Dimension
        K: Number of flow
        """
        with tf.variable_scope('enc_nf_radial', reuse = None):
            w_h = tf.get_variable('w_h', [D, H], initializer=initializer())
            b_h = tf.get_variable('b_h', [H])
            w_mu = tf.get_variable('w_mu', [H, Z], initializer=initializer())
            b_mu = tf.get_variable('b_mu', [Z])
            w_v = tf.get_variable('w_v', [H, Z], initializer=initializer())
            b_v = tf.get_variable('b_v', [Z])
    
            # Weights for outputting normalizing flow parameters
            w_z0s = tf.get_variable('w_z0s', [H, K*Z])
            b_z0s = tf.get_variable('b_z0s', [K*Z])
            w_alphas = tf.get_variable('w_alphas', [H, K])
            b_alphas = tf.get_variable('b_alphas', [K])
            w_betas = tf.get_variable('w_betas', [H, K])
            b_betas = tf.get_variable('b_betas', [K])

    
            h = tf.nn.relu(tf.matmul(x, w_h) + b_h)
            mu = tf.matmul(h, w_mu) + b_mu
            log_var = tf.matmul(h, w_v) + b_v
            # z = mu + tf.sqrt(tf.exp(log_var))*e
    
            # Normalizing Flow parameters
            z0s = tf.matmul(h, w_z0s) + b_z0s
            alphas = tf.matmul(h, w_alphas) + b_alphas
            betas = tf.matmul(h, w_betas) + b_betas
            
            lambd = (z0s, alphas, betas)
        return mu, log_var, lambd

    def decoder(self, z, D, H, Z, initializer=tf.contrib.layers.xavier_initializer, 
                out_fn=tf.nn.sigmoid):
        """
        z: Output from the flow
        D: Input data dimension. 784 in case of MNIST.
        H: Number of hidden units
        Z: Latent dimensions
        """
        with tf.variable_scope('decoder'):
            w_h = tf.get_variable('w_h', [Z, H], initializer=initializer())
            b_h = tf.get_variable('b_h', [H])
            w_mu = tf.get_variable('w_mu', [H, D], initializer=initializer())
            b_mu = tf.get_variable('b_mu', [D])
            w_v = tf.get_variable('w_v', [H, D], initializer=initializer())
            b_v = tf.get_variable('b_v', [D])
            
            print "z in decoder shape", z.get_shape()
            print "w_h in decoder shape", w_h.get_shape()
            print "w_v in decoder shape", w_v.get_shape()
            print "b_h in decoder shape", b_h.get_shape()
            print "b_v in decoder shape", b_v.get_shape()
            h = tf.nn.relu(tf.matmul(z, w_h) + b_h)
            out_mu = tf.matmul(h, w_mu) + b_mu
            out_log_var = tf.matmul(h, w_v) + b_v
            out = out_fn(out_mu)
        return out, out_mu, out_log_var
        