import pdb

import numpy as np
import math
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_kl_divergence_standard
from util import gaussian_logp
from util import gaussian_logp0
from util import bernoulli_logp


class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, temperature, num_maps, num_zsamples,):
        super(VAE, self).__init__(
            # encoder
            qlin0 = L.Linear(dim_in, dim_hidden),
            qlin1 = L.Linear(2*dim_hidden, dim_hidden),
            qlin2 = L.Linear(2*dim_hidden, dim_hidden),
            qlin3 = L.Linear(2*dim_hidden, dim_hidden),
            qlin_mu = L.Linear(2*dim_hidden, dim_latent),
            qlin_ln_var = L.Linear(2*dim_hidden, dim_latent),
            # flow 
            linearity_W = L.Scale(axis=1, W_shape=(dim_latent), bias_term=False),
            linearity_b = L.Bias(axis=0, shape=(1)),
            scaling = L.Scale(axis=1, W_shape=(dim_latent), bias_term=False),
            # decoder
            plin0 = L.Linear(dim_latent, dim_hidden),
            plin1 = L.Linear(2*dim_hidden, dim_hidden),
            plin2 = L.Linear(2*dim_hidden, dim_hidden),
            plin3 = L.Linear(2*dim_hidden, dim_hidden),
            plin_mu = L.Linear(2*dim_hidden, dim_in),
            plin_ber_prob = L.Linear(2*dim_hidden, dim_in),
        )
        self.temperature = temperature
        self.num_zsamples = num_zsamples
        self.epochs_seen  = 0
        # planar mappings
        #cuda.check_cuda_available()
        #xp = cuda.cupy
        # self.planar_maps = []
        # for i in range(num_maps):
            #mapping = {'linearity': L.Linear(dim_latent, dim_latent, nobias=False), 'scaling': L.Scale(axis=1, W_shape=(dim_latent), bias_term=False) }
            # mapping = {'linearity_W': L.Scale(axis=1, W_shape=(dim_latent), bias_term=False), 'linearity_b': L.Bias(axis=0, shape=(1)), \
            #  'scaling': L.Scale(axis=1, W_shape=(dim_latent), bias_term=False) }
            # self.planar_maps.append(mapping) 
            # self.planar_maps[i]['linearity_W'].to_gpu(0)
            # self.planar_maps[i]['linearity_b'].to_gpu(0)
            # self.planar_maps[i]['scaling'].to_gpu(0)

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

        self.p_ber_prob_logit = self.plin_ber_prob(h)

        return self.p_ber_prob_logit

    def __call__(self, x):
        # Compute q(z|x)
        encoding_time = time.time()
        qmu, qln_var = self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        self.logp = 0

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']

        for j in xrange(self.num_zsamples):
            # Sample z ~ q(z_0|x)
            z_0 = F.gaussian(self.qmu, self.qln_var)

            # Perform planar flow mappings, Equation (10)
            decoding_time = time.time()
            h = self.linearity_W(z_0)
            h = F.sum(h,axis=(1))
            h = self.linearity_b(h)
            h = F.tanh(h)
            h_tanh = h # Store for use in ELBO

            dim_latent = z_0.shape[1]
            h = F.transpose(F.tile(h,(dim_latent,1)))
            h = self.scaling(h)

            z_K = z_0 + h

            # Obtain parameters for p(x|z_K)
            p_ber_prob_logit =  self.decode(z_K)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute log q(z_0)
            q_prior_log = gaussian_logp0(z_0)
            
            # Compute log p(x|z_K)
            decoder_log = bernoulli_logp(x, p_ber_prob_logit)
            # Compute log p(z_K)
            p_prior_log = gaussian_logp0(z_K)

            # Compute log p(x,z_K) which is log p(x|z_K) + log p(z_K)
            joint_log = decoder_log + p_prior_log

            # Compute second term of log q(z_K)
            # pdb.set_trace()
            h_tanh_derivative = 1-(h_tanh*h_tanh)
            h_tanh_derivative = F.transpose(F.tile(h_tanh_derivative, (dim_latent,1))) 
            phi = self.linearity_W(h_tanh_derivative) # Equation (11)
            lodget_jacobian = F.sum(self.scaling(phi), axis=1)
            q_K_log = F.log(1 + lodget_jacobian)
            

        decoding_time_average /= self.num_zsamples
        # pdb.set_trace()
        self.obj_batch = ((q_prior_log -joint_log) - q_K_log)
        batch_size = self.obj_batch.shape[0]

        self.obj = F.sum(self.obj_batch)/batch_size

        self.kl = self.obj_batch
        self.logp = self.obj_batch

        self.timing_info = np.array([encoding_time,decoding_time])
        
        return self.obj

