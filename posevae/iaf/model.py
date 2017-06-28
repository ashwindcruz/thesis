import pdb

import math
import numpy as np
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_logp
from util import gaussian_logp0

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, temperature, num_zsamples=1):
        super(VAE, self).__init__(
            # encoder
            qlin0 = L.Linear(dim_in, dim_hidden),
            qlin1 = L.Linear(2*dim_hidden, dim_hidden),
            qlin2 = L.Linear(2*dim_hidden, dim_hidden),
            qlin3 = L.Linear(2*dim_hidden, dim_hidden),
            qlin_mu = L.Linear(2*dim_hidden, dim_latent),
            qlin_ln_var = L.Linear(2*dim_hidden, dim_latent),
            qlin_h = L.Linear(2*dim_hidden, dim_latent),
            # IAF
            qiaf1a = L.Linear(2*dim_latent, dim_latent),
            qiaf1b = L.Linear(2*dim_latent, 2*dim_latent),
            # IAF
            #qiaf2a = L.Linear(2*dim_latent, dim_latent),
            #qiaf2b = L.Linear(2*dim_latent, 2*dim_latent),
            # IAF
            #qiaf3a = L.Linear(2*dim_latent, dim_latent),
            #qiaf3b = L.Linear(2*dim_latent, 2*dim_latent),
            # decoder
            plin0 = L.Linear(dim_latent, dim_hidden),
            plin1 = L.Linear(2*dim_hidden, dim_hidden),
            plin2 = L.Linear(2*dim_hidden, dim_hidden),
            plin3 = L.Linear(2*dim_hidden, dim_hidden),
            plin_mu = L.Linear(2*dim_hidden, dim_in),
            plin_ln_var = L.Linear(2*dim_hidden, dim_in),
        )
        self.num_zsamples = num_zsamples
        self.temperature = temperature
        self.epochs_seen = 0

    def encode(self, x):
        h = F.crelu(self.qlin0(x))
        h = F.crelu(self.qlin1(h))
        h = F.crelu(self.qlin2(h))
        h = F.crelu(self.qlin3(h))

        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)
        self.qh = self.qlin_h(h)

    def decode(self, z):
        h = F.crelu(self.plin0(z))
        h = F.crelu(self.plin1(h))
        h = F.crelu(self.plin2(h))
        h = F.crelu(self.plin3(h))

        self.pmu = self.plin_mu(h)
        self.pln_var = self.plin_ln_var(h)

    def iaf(self, z, h, lin1, lin2):
        ms = F.crelu(lin1(F.concat((z, h), axis=1)))
        ms = lin2(ms)
        m, s = F.split_axis(ms, 2, axis=1)
        s = F.sigmoid(s)
        z = s*z + (1-s)*m
        # pdb.set_trace()
        return z, -F.sum(F.log(s), axis=1)

    def __call__(self, x):
        # Obtain parameters for q(z|x)
        encoding_time = time.time()
        self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        self.logp_xz = 0
        self.logq = 0

        # For reporting purposes only
        self.logp = 0
        self.kl = 0

        decoding_time_average = 0.

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']
        
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            decoding_time = time.time()

            # Apply inverse autoregressive flow (IAF)
            self.logq += gaussian_logp(z, self.qmu, self.qln_var)    # - log q(z|x)

            z, delta_logq1 = self.iaf(z, self.qh, self.qiaf1a, self.qiaf1b)
            #z, delta_logq2 = self.iaf(z, self.qh, self.qiaf2a, self.qiaf2b)
            #z, delta_logq3 = self.iaf(z, self.qh, self.qiaf3a, self.qiaf3b)
            # pdb.set_trace()
            self.logq += delta_logq1 #+ delta_logq2 #+ delta_logq3
            self.logq *= current_temperature

            # Compute p(x|z)
            self.decode(z)

            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute objective, p(x,z)
            logz_given_x = gaussian_logp(x, self.pmu, self.pln_var) # p(x|z)
            logz = (current_temperature*gaussian_logp0(z)) # p(z)
            self.logp_xz += (logz_given_x + logz)

            # For reporting purposes only
            self.logp += logz_given_x
            self.kl += (self.logq - logz) 



        decoding_time_average /= self.num_zsamples
        self.logp_xz /= self.num_zsamples
        self.logq /= self.num_zsamples

        # For reporting purposes only
        self.logp /= self.num_zsamples
        self.kl /= self.num_zsamples


        
        self.obj_batch = self.logp_xz - self.logq     # variational free energy
        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        self.obj = -F.sum(self.obj_batch)/batch_size  

        return self.obj

