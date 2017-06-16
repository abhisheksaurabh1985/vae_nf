import math
import numpy as np
import tensorflow as tf

const = - 0.5 * math.log(2*math.pi)

def log_normal_pdf(x, mean, covariance, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance matrix.
    Parameters
    ----------
    x : 1-D numpy array of len n. It is a single observation at which we want to evaluate pdf.
    mean: 1-D numpy array. Mean of the values along each dimension.
    covariance : Numpy array of size n*n.
    eps : Float. Small number added to the elements in covariance matrix to avoid NaNs.

    Returns
    -------
    _log_normal_pdf: Float
    """
    ndim= len(x)
    if ndim== len(mean) and (ndim, ndim) == covariance.shape:
        determinant_covariance= np.linalg.det(covariance)
        if determinant_covariance== 0:
            raise NameError("Determinant zero. Covariance matrix has to be non-singular")
        covariance += eps

    _log_normal_pdf= ndim*const - 0.5* np.log(np.prod(covariance.diagonal())) \
                  - np.dot(np.dot((x-mean)[np.newaxis],np.linalg.inv(covariance)), (x-mean)[np.newaxis].T)
    return np.asscalar(_log_normal_pdf)
    
def log_normal(x, mean, var, eps=1e-5):
    var += eps
    return const - tf.log(var)/2 - (x - mean)**2 / (2*var)
    

def log_stdnormal(x):
    return const - x**2 / 2

def gaussian_log_pdf(mu, log_var, z):
    """
    Log probability from a diagonal covariance normal distribution.   
    """
    return tf.contrib.distributions.MultivariateNormalDiag(
                loc=mu, scale_diag=tf.maximum(tf.exp(0.5 * log_var), 1e-15)).log_pdf(z)
'''
Test scripts. Comment out the following once tested. 
'''

# Test log_normal_pdf
#x= np.array([3, 4, 5, 2])
#mu= np.array([2.5, 3.5, 4, 1])
#sigma= np.array([[ 2.3,  0. ,  0. ,  0. ],
#       [ 0. ,  1.5,  0. ,  0. ],
#       [ 0. ,  0. ,  1.7,  0. ],
#       [ 0. ,  0. ,  0. ,  2. ]])
#
#eps= 0.000001
#a=log_normal_pdf(x, mu, sigma, eps)
#print a
#print type(a)
#
## Test multivariate standard normal distribution
#x= np.array([3, 4, 5, 2])
#mu= np.array([0,0,0,0])
#sigma= np.array([[ 1,  0. ,  0. ,  0. ],
#       [ 0. ,  1,  0. ,  0. ],
#       [ 0. ,  0. ,  1,  0. ],
#       [ 0. ,  0. ,  0. ,  1 ]])
#eps= 0.000001
#a=log_normal_pdf(x, mu, sigma, eps)
#print a
#print type(a)