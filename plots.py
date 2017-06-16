#from tensorflow.examples.tutorials.mnist import input_data
#mnist= input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import tensorflow as tf

from config import *

def plot_latent_space(X, sess, z_mu, dataset, dataset_name = 'mnist_from_tf'):
    '''
    Plot the two dimensional latent space.
    param dataset_name: Dataset to be used. Defined in the config file. 
    '''
    if dataset_name== 'mnist_from_tf':
        # dataset= mnist
        x_sample, y_sample = dataset.test.next_batch(5000)
        z_mean = sess.run(z_mu, feed_dict={X: x_sample}) # sess.run([z_mu, z_logvar], feed_dict={X: x_sample}) # z_mu has a dimension batch_size*number_latent_dimension.
        plt.figure(figsize=(8, 6)) 
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=np.argmax(y_sample, 1))
        plt.colorbar()
        plt.grid()
        plt.title('2D Latent space')
        # plt.close()
    return 

def plot_reconstruction(X, sess, x_recons, dataset_name = 'mnist_from_tf'):
    '''
    For a given test set, plot the real and the reconstruced image side by side. 
    Right now only for MNIST. 
        param dataset: Use MNIST from tensorflow now. 
    Returns:
        None
    '''
    if dataset_name== 'mnist_from_tf':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        print 'Test run'
        x_sample = mnist.test.next_batch(100)[0] 
        # e_ = np.random.normal(size=(x_sample.shape[0], z_dim))

        # If dataset= mnist, dataset.test.next_batch(batch_size) returns a two tuple.
        # Each element of the tuple is an numpy.ndarray. 
        # mnist.test.next_batch(100)[0].shape is (batch_size, 784) which is the data. 
        # mnist.test.next_batch(100)[1].shape is (batch_size, 10) which is the ground truths. 
        x_reconstruct = sess.run(x_recons, 
                             feed_dict={X: x_sample})
        
        plt.figure(figsize=(8, 12))
        for i in range(5):
            plt.subplot(5, 2, 2*i + 1)
            plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
            plt.title("Test input")
            plt.colorbar()
            plt.subplot(5, 2, 2*i + 2)
            plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.colorbar()
        plt.tight_layout()  
        return

def plot_digits_in_latent_space(sess, prob_z_sample, z_sample, dataset, dataset_name = 'mnist_from_tf', latent_dim = 2):
    '''
    Generate some points on the x and the y axis. 
    For all the (x,y) coordinates 
    Since the prior on the latent is standard Gaussian, transform the points through
    inverse CDF of a Gaussian distribution. Inverse of CDF gives a uniform 
    distribution. 

    '''
    if latent_dim==2:
        if dataset_name== 'mnist_from_tf':
            npoints_x= npoints_y= 15
            digit_size= 28
            canvas= np.empty((digit_size*npoints_x, digit_size*npoints_y))
            linspace_min= 0.15
            linspace_max= 0.95
            x_values= np.linspace(linspace_min, linspace_max, npoints_x) 
            y_values= np.linspace(linspace_min, linspace_max, npoints_y) 
            grid_x = norm.ppf(x_values, loc= 0.0, scale= 1.0)
            grid_y = norm.ppf(y_values, loc= 0.0, scale= 1.0)
            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    z = np.array([[xi, yi]])
                    x_decoded = sess.run(prob_z_sample, feed_dict={z_sample: z})  
                    digit = x_decoded[0].reshape(digit_size, digit_size) 
                    canvas[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit
            plt.figure(figsize=(10, 10))
            plt.imshow(canvas, cmap='Greys_r')
            # plt.show()
    else:
        print "Only 2D latent space plot is available."
        
    return

# Plot loss vs epoch
def plot_loss_vs_epoch(epoch, _loss):
    # plt.gca().set_color_cycle(['red', 'green', 'blue'])
    plt.figure(figsize=(10, 10))
    plt.scatter(range(nepochs), _loss, color= 'b', marker= '*')
    plt.legend(['Avg. Loss'], loc='upper right')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Avg. Loss (in nats)")
    plt.title("Loss vs Epoch (Training)")
    return    

# Plot latent space
def generate_gif(dirname, dataset_name = 'mnist_from_tf', latent_dim = 2):
    if latent_dim==2:
        if dataset_name== 'mnist_from_tf':
            os.system('convert -delay 50 -loop 0 {0}/lspace_mnist-from-tf_*png {0}/lspace_mnist-from-tf.gif'.format(dirname))
    return

def reconstruct(sess, input_data, out_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    x_rec = sess.run([out_op], feed_dict={x_op: input_data, e_op: e_})
    return x_rec

def show_reconstruction(actual, recon):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(actual.reshape(28, 28), cmap='gray')
    axs[1].imshow(recon.reshape(28, 28), cmap='gray')
    axs[0].set_title('actual')
    axs[1].set_title('reconstructed')
    plt.show()

