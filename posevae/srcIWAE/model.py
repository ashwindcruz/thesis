
import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
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

def gaussian_lik(x, mu, ln_var):
    """log N(x ; mu, var)"""
    batchsize = mu.data.shape[0]
    D = x.data.size
    S = F.exp(ln_var)
    xc = x - mu

    logp_sum = -0.5*(F.sum((xc*xc) / S) + F.sum(ln_var)
        + D*math.log(2.0*math.pi))

    return F.exp(logp_sum / batchsize)

def logsumexp(collected):
    # Using streaming method proposed from:
    # http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
     
    minimum = np.inf
    for var in collected:
        value = float(var.data)
        if(value<minimum):
            minimum = value
        
    minimum = chainer.Variable(cuda.cupy.asarray(minimum, dtype=np.float32))

    summedVal = 0        
    for var in collected:
        summedVal += F.exp(var - minimum)        

    return(minimum + F.log(summedVal))

    #minimum = chainer.Variable(cuda.cupy.asarray(np.inf, dtype=np.float32))  
    #r_val = 0.0
    #for var in collected:
    #    if float(var.data)>=float(minimum.data):
    #        r_val += F.exp(var - minimum)
    #        pdb.set_trace()
    #    else:
    #        pdb.set_trace()
    #        r_val *= F.exp(minimum - var)
    #        r_val += 1.0
    #        pdb.set_trace()
    #        minimum = var
    #pdb.set_trace()
    #return(F.log(r) + minimum)

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
            
            # Compute w' for this sample
            k_log = chainer.Variable(xp.asarray(np.log(self.num_zsamples), dtype=np.float32))
            
            # Store the latest log weight'
            self.w_holder.append(decoder_log + prior_log - encoder_log - k_log)
        
        self.pre_obj = logsumexp(self.w_holder)

        self.obj = self.pre_obj*-1 
        return self.obj

