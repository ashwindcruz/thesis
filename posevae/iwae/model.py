import pdb

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_logp
from util import logsumexp


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
        # Obtain parameters for q(z|x)
        self.encode(x)
        xp = cuda.cupy
        self.importance_weights = 0
        self.w_holder = []

        for j in xrange(self.num_zsamples):
            # Sample z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            # Compute log q(z|x)
            encoder_log = gaussian_logp(z, self.qmu, self.qln_var)
            
            # Obtain parameters for p(x|z)
            self.decode(z)

            # Compute log p(x|z)
            decoder_log = gaussian_logp(x, self.pmu, self.pln_var)
            
            # Compute log p(z). The odd notation being used is to supply a mean of 0 and covariance of 1
            prior_log = gaussian_logp(z, self.qmu*0, self.qln_var/self.qln_var)
            
            # Compute w' for this sample (batch)
            #k_log = chainer.Variable(xp.asarray(np.log(self.num_zsamples), dtype=np.float32))
            k_log = np.log(self.num_zsamples)
            # pdb.set_trace()

            # Store the latest log weight'
            self.w_holder.append(decoder_log + prior_log - encoder_log - k_log)
        # pdb.set_trace()

        # Collate the information from the various samples
        # self.obj = []# np.zeros([1,8192])
        # for i in range(8192):
            # pdb.set_trace()
        # self.obj.append(logsumexp(self.w_holder)
        # self.obj = logsumexp(self.w_holder)
        # pdb.set_trace()
        #self.obj = chainer.Variable(self.obj)

        self.obj = logsumexp(self.w_holder)
        # pdb.set_trace()
        return -self.obj

