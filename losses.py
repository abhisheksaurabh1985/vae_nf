import tensorflow as tf
from distributions import *

def cross_entropy_loss(prediction, actual, offset= 1e-4):
    """
    param prediction: tensor of observed values
    param actual: tensor of actual values
    """
    with tf.name_scope("cross_entropy"):
#        predicate = tf.logical_or(tf.less(1e-10 + prediction, 1e-7), 
#                                  tf.less(1e-10 + 1 + prediction, 1e-7))
#        fn1 = - tf.reduce_sum(actual * tf.log(1e-8)
#                         + (1 - actual) * tf.log(1e-8), 1)
#        fn2 = - tf.reduce_sum(actual * tf.log(1e-10 + prediction)
#                             + (1 - actual) * tf.log(1e-10 + 1 - prediction), 1)
#        ce_loss = tf.cond(predicate, lambda: fn1, lambda: fn2)
        _prediction = tf.clip_by_value(prediction, offset, 1 - offset)
        ce_loss= - tf.reduce_sum(actual * tf.log(_prediction)
                             + (1 - actual) * tf.log(1 - _prediction), 1)
#        ce_loss= - tf.reduce_sum(actual * tf.log(1e-10 + prediction)
#                             + (1 - actual) * tf.log(1e-10 + 1 - prediction), 1)
        return ce_loss
                              
def kl_divergence_gaussian(mu, var):
    with tf.name_scope("kl_divergence"):
        # kl = - 0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1)
        _var = tf.clip_by_value(var, 1e-4, 1e6)
        kl = - 0.5 * tf.reduce_sum(1 + tf.log(_var) - tf.square(mu) - \
                                   tf.exp(tf.log(_var)), 1)    
        return kl 
    
def gaussian_log_pdf(z, mu, var):
    """
    Log probability from a diagonal covariance normal distribution.   
    """
    return tf.contrib.distributions.MultivariateNormalDiag(
                loc = mu, scale_diag = tf.maximum(tf.sqrt(var), 1e-4)).log_prob(z + 1e-4)
    
def elbo_loss(actual, prediction, beta, global_step, recons_error_func = cross_entropy_loss, **kwargs):
#    monitor = {}
    mu= kwargs['z_mu']
    _var = kwargs['z_var']
    
    if 'logdet_jacobian'not in kwargs:
        # In the paper, kl_loss is negative and recons_loss (i.e. the cross_entropy_loss) is positive.
        # The kl_divergence_gaussian function returns a negative quantity. 
        # This part seems to be correct as far as the sign is concerned. 
        kl_loss = kl_divergence_gaussian(mu, _var)
        # kl_loss = tf.reduce_mean(kl_divergence_gaussian(mu, tf.log(_var)), name ='kl') # kl_div function expects log variance.
        recons_loss = recons_error_func(prediction, actual)
        # recons_loss = tf.reduce_mean(recons_error_func(prediction, actual), 
        #                             name = 'recons_loss')
        _elbo_loss = tf.reduce_mean(recons_loss + kl_loss)
        return _elbo_loss # , recons_loss, kl_loss                             
    else:
        # First term is +ve. Rest all terms are negative. 
        z0 = kwargs['z0']
        zk = kwargs['zk']
        logdet_jacobian = kwargs['logdet_jacobian']
       
        # First term                                     
        log_q0_z0 = gaussian_log_pdf(z0, mu, _var)
        # Third term
        # sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian, 
        #                                     name='sum_logdet_jacobian')
        sum_logdet_jacobian = logdet_jacobian
        # First term - Third term                           
        log_qk_zk = log_q0_z0 - sum_logdet_jacobian

        # First component of the second term: p(x|z_k)  
        if beta:
            beta_t = tf.minimum(1.0, 0.01 + tf.cast(global_step/10000, tf.float32)) # global_step
            log_p_x_given_zk =  beta_t * recons_error_func(prediction, actual)
            log_p_zk =  beta_t * gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu))
        else:
            log_p_x_given_zk =  recons_error_func(prediction, actual)
            log_p_zk =  gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu))
        
        recons_loss =  log_p_x_given_zk
        kl_loss = log_qk_zk - log_p_zk  
        _elbo_loss = tf.reduce_mean(kl_loss + recons_loss)
        return _elbo_loss    
        
def make_loss(pred, actual, log_var, mu, log_detj, z0, sigma=1.0):
    """
    NOT USING
    """
    kl = -tf.reduce_mean(0.5*tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1))
    offset = 1e-7 
    prediction_ = tf.clip_by_value(pred, offset, 1 - offset)
    cross_entropy_loss = tf.reduce_mean(actual * tf.log(prediction_) + (1 - actual) * tf.log(1 - prediction_), 1)
    # rec_err = 0.5*(tf.nn.l2_loss(actual - pred)) / sigma
    loss = tf.reduce_mean(kl + cross_entropy_loss - log_detj)
    return loss

def vanilla_vae_loss(x, x_reconstr_mean, z_mu, z_var):
    reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean)
                       + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1) 
    latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(z_var) - tf.square(z_mu) - \
                         tf.exp(tf.log(z_var)), 1)
    cost = tf.reduce_mean(reconstr_loss + latent_loss)
    return cost

def log_normal(x, mean, var, eps=1e-5):
    const = - 0.5 * tf.log(2*math.pi)
    var += eps
    return const - tf.log(var)/2 - (x - mean)**2 / (2*var)
