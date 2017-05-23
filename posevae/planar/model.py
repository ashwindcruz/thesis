#import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import cupy
#import cuda.cupy as xp

import pdb

def gaussian_kl_divergence(mu, ln_var):
    """D_{KL}(N(mu,var) | N(0,1))"""
    batchsize = mu.data.shape[0]
    S = F.exp(ln_var)
    D = mu.data.size

    KL_sum = 0.5*(F.sum(S) + F.sum(mu*mu) - F.sum(ln_var) - D)

    return KL_sum / batchsize

def gaussian_logp(x, mu, ln_var):
    """log N(x ; mu, var)"""
    batchsize = mu.data.shape[0]
    D = x.data.size
    S = F.exp(ln_var)
    xc = x - mu

    logp_sum = -0.5*(F.sum((xc*xc) / S) + F.sum(ln_var)
        + D*math.log(2.0*math.pi))

    return logp_sum / batchsize

xp = cuda.cupy

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_zsamples, num_maps, batchsize):
        super(VAE, self).__init__(
            # encoder
            qlin0 = L.Linear(dim_in, dim_hidden),
            qlin1 = L.Linear(2*dim_hidden, dim_hidden),
            qlin2 = L.Linear(2*dim_hidden, dim_hidden),
            qlin3 = L.Linear(2*dim_hidden, dim_hidden),
            qlin_mu = L.Linear(2*dim_hidden, dim_latent),
            qlin_ln_var = L.Linear(2*dim_hidden, dim_latent),
            # decoder
            plin0 = L.Linear(dim_latent, dim_hidden),
            plin1 = L.Linear(2*dim_hidden, dim_hidden),
            plin2 = L.Linear(2*dim_hidden, dim_hidden),
            plin3 = L.Linear(2*dim_hidden, dim_hidden),
            plin_mu = L.Linear(2*dim_hidden, dim_in),
            plin_ln_var = L.Linear(2*dim_hidden, dim_in),
        )
        self.num_zsamples = num_zsamples
        # planar mappings
        #cuda.check_cuda_available()
        #xp = cuda.cupy
        self.planar_maps = []
        for i in range(num_maps):
            #mapping = {'linearity': L.Linear(dim_latent, dim_latent, nobias=False), 'scaling': L.Scale(axis=1, W_shape=(dim_latent), bias_term=False) }
            mapping = {'linearity_W': L.Scale(axis=1, W_shape=(dim_latent), bias_term=False), 'linearity_b': L.Bias(axis=0, shape=(1)), \
             'scaling': L.Scale(axis=1, W_shape=(dim_latent), bias_term=False) }
            self.planar_maps.append(mapping) 
            self.planar_maps[i]['linearity_W'].to_gpu(0)
            self.planar_maps[i]['linearity_b'].to_gpu(0)
            self.planar_maps[i]['scaling'].to_gpu(0)

            #pdb.set_trace()
    def encode(self, x):
        h = F.crelu(self.qlin0(x))
        h = F.crelu(self.qlin1(h))
        h = F.crelu(self.qlin2(h))
        h = F.crelu(self.qlin3(h))

        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)
        #pdb.set_trace()
        return self.qmu, self.qln_var

    def planar_flows(self,z):
        self.z_trans = []
        self.z_trans.append(z)

        for i in range(len(self.planar_maps)):
            
            # Relu non-linearity
            #h = F.relu(self.planar_maps[i]['linearity'](z))
            #h.grad = xp.ones(h.shape, dtype=xp.float32)
            #h.backward()
            #h.cleargrads()
            #pdb.set_trace()
            
            # Sigmoid non-linearity
            h = self.planar_maps[i]['linearity_W'](z)
            h = F.sum(h,axis=(1))
            h = self.planar_maps[i]['linearity_b'](h)
            h = F.sigmoid(h)
            h_sigmoid = h
            dim_latent = z.shape[1]
            h = F.transpose(F.tile(h,(dim_latent,1)))
            h = self.planar_maps[i]['scaling'](h)

            # Store the lodget-Jacobian terms for later ELBO calculation
            #h_derivative = F.sigmoid(self.planar_maps[i]['linearity'](z)) * (1 - F.sigmoid(self.planar_maps[i]['linearity'](z)))
            h_derivative = h_sigmoid*(1-h_sigmoid)
            h_derivative = F.transpose(F.tile(h_derivative, (dim_latent,1)))    

            self.planar_maps[i]['lodget_jacobian'] = self.planar_maps[i]['linearity_W'](h_derivative) #TODO: You need to have the weights of the linearity here
            z += h
            self.z_trans.append(z)
        return z

    def decode(self, z):
        h = F.crelu(self.plin0(z))
        h = F.crelu(self.plin1(h))
        h = F.crelu(self.plin2(h))
        h = F.crelu(self.plin3(h))

        self.pmu = self.plin_mu(h)
        self.pln_var = self.plin_ln_var(h)

        return self.pmu, self.pln_var

    def __call__(self, x):
        # Compute q(z|x)
        qmu, qln_var = self.encode(x)
        #xp = cuda.cupy
        self.kl = gaussian_kl_divergence(qmu, qln_var)
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # Sample z ~ q(z_0|x)
            z_0 = F.gaussian(self.qmu, self.qln_var)

            # Perform planar flow mappings, Equation (10)
            z_K = self.planar_flows(z_0)

            # Obtain parameters for p(x|z_K)
            pmu, pln_var =  self.decode(z_K)

            # Compute log q(z_0)
            q_prior_log = gaussian_logp(z_0, qmu*0, qln_var/qln_var)

            # Compute log p(x,z_K)
            # Compute log p(x|z_K)
            decoder_log = gaussian_logp(x, pmu, pln_var)
            # Compute log p(z_K)
            p_prior_log = gaussian_logp(z_K, qmu*0, qln_var/qln_var)
            joint_log = decoder_log + p_prior_log

            # Compute second term of log q(z_K)
            trans_log = 0
            for k in range(len(self.z_trans)-1):
                #pdb.set_trace()
                lodget_jacobian_scaled = self.planar_maps[k]['scaling'](self.planar_maps[k]['lodget_jacobian'])
                lodget_jacobian_scaled = F.sum(lodget_jacobian_scaled, axis=1)
                #pdb.set_trace()
                trans_log += F.log(1+lodget_jacobian_scaled)

            #self.logp += gaussian_logp(x, self.pmu, self.pln_var)

        #self.logp /= self.num_zsamples
        #self.obj = self.kl - self.logp
        batch_size = trans_log.shape[0]
        trans_log = F.sum(trans_log, axis=0)/batch_size
        #pdb.set_trace()
        self.obj = -((q_prior_log -joint_log) - trans_log)
        print(self.obj.data)
        return self.obj

