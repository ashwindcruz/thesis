import pdb

import numpy as np
import math
import subprocess
from subprocess import PIPE

import chainer
from chainer import cuda
from chainer.cuda import cupy
import chainer.functions as F
#import cuda

def print_compute_graph(file, g):
    format = file.split('.')[-1]
    cmd = 'dot -T%s -o %s'%(format,file)
    p=subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    p.stdin.write(g.dump())
    p.communicate()
    return p.returncode

def gaussian_kl_divergence_standard(mu, ln_var):
   """D_{KL}(N(mu,var) | N(0,1))"""
   batch_size = float(mu.data.shape[0])
   S = F.exp(ln_var)
   D = mu.data.size
   
   KL_sum = 0.5*(F.sum(S, axis=1) + F.sum(mu*mu, axis=1) - F.sum(ln_var, axis=1) - D/batch_size)

   return KL_sum #/ batchsize

def gaussian_logp(x, mu, ln_var):
    """log N(x ; mu, var)"""
    batch_size = mu.data.shape[0]
    D = x.data.size
    S = F.exp(ln_var)
    xc = x - mu

    logp_sum = -0.5*(F.sum((xc*xc) / S, axis=1) + F.sum(ln_var, axis=1)
        + D/batch_size*math.log(2.0*math.pi))


    return logp_sum #/ batchsize

def gaussian_kl_divergence(z_0, z_0_mu, z_0_ln_var, z_T):
    """D_{KL}(q(z_0|x) || p(z_T))"""
    logp_q = gaussian_logp(z_0, z_0_mu, z_0_ln_var)
    logp_p = gaussian_logp(z_T, z_0_mu*0, z_0_ln_var/z_0_ln_var)
    kl_loss = logp_q - logp_p

    return kl_loss

def logsumexp(collected):
# Using streaming method proposed from:
# http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
  # pdb.set_trace()
  xp = cuda.cupy
  # batch_size = collected[0].shape[0]
  # minimum = collected[0]#np.inf*chainer.Variable(xp.ones([batch_size],dtype='float32'))
  # for var in collected:
  #     # value = float(var.data)
  #     less_than = xp.less(var.data,minimum.data)
  #     minimum.data = xp.where(less_than, var.data, minimum.data)
  #     # if(value<minimum):
  #     #     minimum = value
  # # pdb.set_trace()    
  # # if(len([x.data for x in minimum if x.data>0])):
  # #   pdb.set_trace()

  # summedVal = 0        
  # for var in collected:
  #     summedVal += F.exp(var - minimum)        
  # # pdb.set_trace()
  # result = minimum + F.log(summedVal)
  # if(len([x.data for x in result if x.data>0])):
  #   pdb.set_trace()

  average = 0
  for var in collected:
    average += var
  average /= len(collected)

  summedVal = 0
  for var in collected:
    summedVal += F.exp(var - average)

  result = average + F.log(summedVal)
  # if(len([x.data for x in result if x.data>0])):
  #   pdb.set_trace()



  return result

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

# Before changing logsumexp to work with batches instead of single averages
# def logsumexp(collected):
#     # Using streaming method proposed from:
#     # http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
     
#     minimum = np.inf
#     for var in collected:
#         value = float(var.data)
#         if(value<minimum):
#             minimum = value
        
#     minimum = chainer.Variable(cuda.cupy.asarray(minimum, dtype=np.float32))

#     summedVal = 0        
#     for var in collected:
#         summedVal += F.exp(var - minimum)        

#     return(minimum + F.log(summedVal))



