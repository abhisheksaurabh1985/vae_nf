import time
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *
from plots import *

def train_norm_flow(n_samples, latent_space, logfile, z_mu, z_var, sess, 
                    solver, X, nf_free_energy_bound, dataset):
    """
    nf_free_energy_bound: loss_optimizer
    """                    

    avg_free_energy_bound= []
    # Training 
    start_time= time.time() # Record the run time of experiment
    print "###### Training starts ######"
    for epoch in range(nepochs):
        avg_cost= 0
        total_batch= int(n_samples/ batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_xs_labels = dataset.train.next_batch(batch_size)
            # iteration = (epoch + 1) * i
            # print(batch_xs.min(), batch_xs.max())
            _, cost= sess.run([solver, nf_free_energy_bound], feed_dict={X: batch_xs})
            # cost= sess.run([solver, nf_free_energy_bound], feed_dict={X: batch_xs})
            # Compute average loss per epoch
            avg_cost += (cost/ n_samples) *  batch_size
    
        # Store the average values of losses, density functions and logdet_jacobian in list.         
        avg_free_energy_bound.append(avg_cost)
    
        # Display logs per epoch step
        if epoch % display_step == 0:
#            if z_dim==2:
#                # pass
#                # figure()
#                plot_latent_space(X, sess, z_mu, z_var, dataset, dataset_name) # At the end of each epoch plot and save the latent space. 
#                plt.savefig(os.path.join(latent_space,'lspace_' + dataset_name \
#                            + '_%d.png' % epoch))
#                plt.close()
            line =  "Epoch: %i \t Average cost: %0.9f" % (epoch, avg_free_energy_bound[epoch])
            print line
            with open(logfile,'a') as f:
                f.write(line + "\n")
            # samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})
    print("--- %s seconds ---" % (time.time() - start_time))    
    return avg_free_energy_bound


