#!/usr/bin/env python

"""Interpolate two random pose vectors using a learned VAE model.

Author: Sebastian Nowozin <senowozi@microsoft.com>
Date: 5th August 2016

Usage:
  interpolate.py (-h | --help)
  interpolate.py [options] <modelprefix> <output.mat>

Options:
  -h --help                    Show this help screen.
  -g <device>, --device        GPU id to train model on.  Use -1 for CPU [default: -1].
  -s <steps>, --steps          Number of interpolation samples [default: 100].
  -n <samples>, --count        Number of pathes to sample [default: 32].

The modelprefix.meta.yaml file must exist.
"""

import time
import yaml
import numpy as np
import scipy.io as sio
from docopt import docopt

import chainer
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

import model
import util

args = docopt(__doc__, version='interpolate 0.1')
print(args)

model_file = args['<modelprefix>']+'.h5'
yaml_file = args['<modelprefix>']+'.meta.yaml'

with open(yaml_file, 'r') as f:
    argsy = yaml.load(f)

d = 60
nhidden = int(argsy['--nhidden'])
nlatent = int(argsy['--nlatent'])
zcount = int(argsy['--vae-samples'])

vae = model.VAE(d, nhidden, nlatent, zcount)
serializers.load_hdf5(model_file, vae)

gpu_id = int(args['--device'])
print "Running on GPU %d" % gpu_id
if gpu_id >= 0:
    cuda.check_cuda_available()
if gpu_id >= 0:
    xp = cuda.cupy
    vae.to_gpu(gpu_id)
else:
    xp = np

steps = int(args['--steps'])
print "Using %d interpolation steps" % steps
nsamples = int(args['--count'])
print "Sampling %d pathes" % nsamples

# Interpolate
Xsamples = {}
for i in xrange(nsamples):
    with cupy.cuda.Device(gpu_id):
        z1 = np.random.normal(loc=0.0, scale=1.0, size=(1,nlatent))
        z2 = np.random.normal(loc=0.0, scale=1.0, size=(1,nlatent))
        z = np.zeros((steps,nlatent))
        for si in xrange(steps):
            alpha = 1 - si/float(steps-1)
            z[si,:] = alpha*z1 + (1-alpha)*z2
        z = chainer.Variable(xp.asarray(z, dtype=np.float32))
        vae.decode(z)
        #Xsample = F.gaussian(vae.pmu, vae.pln_var)
        Xsample = vae.pmu
        Xsample.to_cpu()
        Xsamples["X_%d" % i] = Xsample.data

sio.savemat(args['<output.mat>'], Xsamples)

