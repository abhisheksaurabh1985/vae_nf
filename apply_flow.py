import tensorflow as tf

from config import *
from NormalizingRadialFlow import NormalizingRadialFlow
from NormalizingPlanarFlow import NormalizingPlanarFlow

def apply_flow(z_sample, flow_params):
    """
    z_sample: z obtained after reparametrizing. 
    flow_type: Defined in configuration.
    batch_size: Defined in configuration. 
    nFlows: Defined in configuration. 
    z_score_normalization: Defined in configuration. 
    """
    z_K = [] # 'Tensor' object does not support item assignment. Hence, store z_K and logdet_jacobian in a list.
    logdet_jacobian= []
    z_K.append(z_sample) # Store the output of the first sampling in the 0th element.
    # logdet_jacobian.append(tf.zeros_like(tf.placeholder(tf.float32, shape=[None, 1])))
    logdet_jacobian.append(tf.zeros([batch_size, 1], tf.float32))
    
    for k in range(nFlows):
        if k==0:
            pass
        else:
            if z_score_normalization== True:
                # z-score standardization
                _mean, _variance= tf.nn.moments(z_K[k-1],axes=[0])
                z_K[k-1]= (z_K[k-1] - _mean) / _variance
                if flow_type== 'Radial':
                    currentClass = NormalizingRadialFlow(z_K[k-1])
                    z_K.append(currentClass.radial_flow()[0])
                    logdet_jacobian.append(currentClass.radial_flow()[1])
                elif flow_type== 'Planar':
                    us, ws, bs = flow_params
                    u, w, b = us[:, k*z_dim:(k+1)*z_dim], ws[:, k*z_dim:(k+1)*z_dim], bs[:, k]
                    print "z_k", z_K[k-1].get_shape()                    
                    currentClass = NormalizingPlanarFlow(z_K[k-1])
                    temp1, temp2 = currentClass.planar_flow_2(u,w,b)
                    z_K.append(temp1)
                    logdet_jacobian.append(temp2)
            else: 
                if flow_type== 'Radial':
                    currentClass = NormalizingRadialFlow(z_K[k-1])
                    z_K.append(currentClass.radial_flow()[0])
                    logdet_jacobian.append(currentClass.radial_flow()[1])
                elif flow_type== 'Planar':
                    us, ws, bs = flow_params
                    u, w, b = us[:, k*z_dim:(k+1)*z_dim], ws[:, k*z_dim:(k+1)*z_dim], bs[:, k]
                    print "z_k", z_K[k-1].get_shape()                    
                    currentClass = NormalizingPlanarFlow(z_K[k-1])
                    temp1, temp2 = currentClass.planar_flow_2(u,w,b)
                    z_K.append(temp1)
                    logdet_jacobian.append(temp2)
    return z_K, logdet_jacobian                