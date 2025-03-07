import pdb

import numpy as np
import math
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_logp
from util import gaussian_logp0
from util import bernoulli_logp

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_zsamples=1):
        super(VAE, self).__init__(
            # encoder
            qlin0 = L.Linear(dim_in, dim_hidden),
            qlin1 = L.Linear(2*dim_hidden, dim_hidden),
            # qlin2 = L.Linear(2*dim_hidden, dim_hidden),
            # qlin3 = L.Linear(2*dim_hidden, dim_hidden),
            qlin_mu = L.Linear(2*dim_hidden, dim_latent),
            qlin_ln_var = L.Linear(2*dim_hidden, dim_latent),
            qlin_h = L.Linear(2*dim_hidden, dim_latent),
            # IAF
            qiaf1a = L.Linear(2*dim_latent, dim_latent),
            qiaf1b = L.Linear(2*dim_latent, 2*dim_latent),
            # IAF
            qiaf2a = L.Linear(2*dim_latent, dim_latent),
            qiaf2b = L.Linear(2*dim_latent, 2*dim_latent),
            # IAF
            qiaf3a = L.Linear(2*dim_latent, dim_latent),
            qiaf3b = L.Linear(2*dim_latent, 2*dim_latent),
            # decoder
            plin0 = L.Linear(dim_latent, dim_hidden),
            plin1 = L.Linear(2*dim_hidden, dim_hidden),
            # plin2 = L.Linear(2*dim_hidden, dim_hidden),
            # plin3 = L.Linear(2*dim_hidden, dim_hidden),
            plin_ber_prob = L.Linear(2*dim_hidden, dim_in),
        )
        self.num_zsamples = num_zsamples
        self.epochs_seen = 0

    def encode(self, x):
        h = F.crelu(self.qlin0(x))
        h = F.crelu(self.qlin1(h))
        # h = F.crelu(self.qlin2(h))
        # h = F.crelu(self.qlin3(h))

        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)
        self.qh = self.qlin_h(h)

    def decode(self, z):
        h = F.crelu(self.plin0(z))
        h = F.crelu(self.plin1(h))
        # h = F.crelu(self.plin2(h))
        # h = F.crelu(self.plin3(h))

        self.p_ber_prob_logit = self.plin_ber_prob(h)

    def iaf(self, z, h, lin1, lin2):
        ms = F.crelu(lin1(F.concat((z, h), axis=1)))
        ms = lin2(ms)
        m, s = F.split_axis(ms, 2, axis=1)
        s = F.sigmoid(s)
        z = s*z + (1-s)*m
        return z, -F.sum(F.log(s), axis=1)

    def __call__(self, x):
        # Compute q(z|x)
        self.encode(x)

        self.logp = 0
        self.logq = 0
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            # Apply inverse autoregressive flow (IAF)
            self.logq += gaussian_logp(z, self.qmu, self.qln_var)    # - log q(z|x)

            z, delta_logq1 = self.iaf(z, self.qh, self.qiaf1a, self.qiaf1b)
            z, delta_logq2 = self.iaf(z, self.qh, self.qiaf2a, self.qiaf2b)
            z, delta_logq3 = self.iaf(z, self.qh, self.qiaf3a, self.qiaf3b)
            self.logq += delta_logq1 + delta_logq2 + delta_logq3

            # Compute p(x|z)
            self.decode(z)

            # Compute objective
            self.logp += bernoulli_logp(x, self.p_ber_prob_logit)
            self.logp += gaussian_logp0(z)

        self.logp /= self.num_zsamples
        self.logq /= self.num_zsamples
        self.obj_batch = -(self.logq - self.logp)    # variational free energy

        # Placeholders just to have the code working
        self.logp = self.obj_batch
        self.kl = self.obj_batch
        self.timing_info = np.array([0.,0.])

        batch_size = self.obj_batch.shape[0]
        self.obj = -F.sum(self.obj_batch)/batch_size

        return self.obj

