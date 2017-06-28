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


class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, temperature, num_zsamples=1):
        # pdb.set_trace()
        # super(VAE, self).__init__(
            # encoder
            # pdb.set_trace()
        super(VAE, self).__init__()
        self.qlin0 = L.Linear(dim_in, dim_hidden)
        self.qlin1 = L.Linear(2*dim_hidden, dim_hidden)
        self.qlin2 = L.Linear(2*dim_hidden, dim_hidden)
        self.qlin3 = L.Linear(2*dim_hidden, dim_hidden)
        self.qlin_mu = L.Linear(2*dim_hidden, dim_latent)
        self.qlin_ln_var = L.Linear(2*dim_hidden, dim_latent)
        # decoder
        self.plin0 = L.Linear(dim_latent, dim_hidden)
        self.plin1 = L.Linear(2*dim_hidden, dim_hidden)
        self.plin2 = L.Linear(2*dim_hidden, dim_hidden)
        self.plin3 = L.Linear(2*dim_hidden, dim_hidden)
        self.plin_mu = L.Linear(2*dim_hidden, dim_in)
        self.plin_ln_var = L.Linear(2*dim_hidden, dim_in)
        # )
        self._children = ['qlin3', 'qlin2', 'qlin1', 'qlin0', 'plin_mu', 'plin_ln_var', 'qlin_mu', 'qlin_ln_var', 'plin2', 'plin3', 'plin0', 'plin1']
        # pdb.set_trace()
        self.temperature = temperature
        self.num_zsamples = num_zsamples
        self.epochs_seen = 0

    def encode(self, x):
        h = F.crelu(self.qlin0(x))
        h = F.crelu(self.qlin1(h))
        h = F.crelu(self.qlin2(h))
        h = F.crelu(self.qlin3(h))

        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)

    def decode(self, z):
        h = F.crelu(self.plin0(z))
        h = F.crelu(self.plin1(h))
        h = F.crelu(self.plin2(h))
        h = F.crelu(self.plin3(h))

        self.pmu = self.plin_mu(h)
        self.pln_var = self.plin_ln_var(h)

    def __call__(self, x):
        # Compute q(z|x)
        encoding_time = time.time()
        self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        self.kl = gaussian_kl_divergence_standard(self.qmu, self.qln_var)
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            # Compute p(x|z)
            decoding_time = time.time()
            self.decode(z)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute objective
            self.logp += gaussian_logp(x, self.pmu, self.pln_var)

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']

        decoding_time_average /= self.num_zsamples
        self.logp /= self.num_zsamples
        self.obj_batch = self.logp - (current_temperature*self.kl)
        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        
        self.obj = -F.sum(self.obj_batch)/batch_size
        
        return self.obj

