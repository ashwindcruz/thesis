
import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

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

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_zsamples=1):
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
        self.encode(x)

        self.kl = gaussian_kl_divergence(self.qmu, self.qln_var)
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            # Compute p(x|z)
            self.decode(z)

            # Compute objective
            self.logp += gaussian_logp(x, self.pmu, self.pln_var)

        self.logp /= self.num_zsamples
        self.obj = self.kl - self.logp

        return self.obj

