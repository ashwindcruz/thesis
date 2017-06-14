import pdb

import math
import numpy as np
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_logp
from util import gaussian_kl_divergence


class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_zsamples=1, house_degree=1):
        super(VAE, self).__init__(
            # encoder
            qlin0 = L.Linear(dim_in, dim_hidden),
            qlin1 = L.Linear(2*dim_hidden, dim_hidden),
            qlin2 = L.Linear(2*dim_hidden, dim_hidden),
            qlin3 = L.Linear(2*dim_hidden, dim_hidden),
            qlin_mu = L.Linear(2*dim_hidden, dim_latent),
            qlin_ln_var = L.Linear(2*dim_hidden, dim_latent),
            qlin_h_vec_0 = L.Linear(2*dim_hidden, dim_latent, initial_bias=-5),
            # decoder
            plin0 = L.Linear(dim_latent, dim_hidden),
            plin1 = L.Linear(2*dim_hidden, dim_hidden),
            plin2 = L.Linear(2*dim_hidden, dim_hidden),
            plin3 = L.Linear(2*dim_hidden, dim_hidden),
            plin_mu = L.Linear(2*dim_hidden, dim_in),
            plin_ln_var = L.Linear(2*dim_hidden, dim_in),
            # linear layer required for v_t of Householder flow transformations
            qlin_h_vec_t = L.Linear(dim_latent, dim_latent, initial_bias=-5),
        )
        self.num_zsamples = num_zsamples
        self.house_degree = house_degree

    def encode(self, x):
        h = F.crelu(self.qlin0(x))
        h = F.crelu(self.qlin1(h))
        h = F.crelu(self.qlin2(h))
        h = F.crelu(self.qlin3(h))

        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)
        self.qh_vec_0 = self.qlin_h_vec_0(h)

        return self.qmu, self.qln_var, self.qh_vec_0

    def house_transform(self,z):
        vec_t = self.qh_vec_0
        
        for i in range(self.house_degree):
            vec_t = F.identity(self.qlin_h_vec_t(vec_t))
            vec_t_product = F.matmul(vec_t, vec_t, transb=True)
            vec_t_norm_sqr = F.tile(F.sum(F.square(vec_t)), (z.shape[0], z.shape[1]))
            z = z - 2*F.matmul(vec_t_product,  z)/vec_t_norm_sqr
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
        # Obtain parameters for q(z|x)
        encoding_time = time.time()
        qmu, qln_var, qh_vec_0 = self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        self.kl = 0
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # z_0 ~ q(z|x)
            z_0 = F.gaussian(qmu, qln_var)

            # Perform Householder flow transformation, Equation (8)
            decoding_time = time.time()
            z_T = self.house_transform(z_0)

            # Obtain parameters for p(x|z_T)
            pmu, pln_var = self.decode(z_T)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute objective
            self.logp += gaussian_logp(x, self.pmu, self.pln_var)
            self.kl += gaussian_kl_divergence(z_0, qmu, qln_var, z_T)
            
        
        decoding_time_average /= self.num_zsamples
        
        self.logp /= self.num_zsamples
        self.kl /= self.num_zsamples
        self.obj_batch = self.logp - self.kl

        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        
        self.obj = -F.sum(self.obj_batch)/batch_size
        
        return self.obj

