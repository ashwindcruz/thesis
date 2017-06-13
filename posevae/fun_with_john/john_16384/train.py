#!/usr/bin/env python

"""Train variational autoencoder (VAE) model for pose data.

Author: Sebastian Nowozin <senowozi@microsoft.com>
Date: 5th August 2016

Usage:
  train.py (-h | --help)
  train.py [options] <data.mat>

Options:
  -h --help                    Show this help screen.
  -g <device>, --device        GPU id to train model on.  Use -1 for CPU [default: -1].
  -o <modelprefix>             Write trained model to given file.h5 [default: output].
  --vis <graph.ext>            Visualize computation graph.
  -b <batchsize>, --batchsize  Minibatch size [default: 100].
  -t <runtime>, --runtime      Total training runtime in seconds [default: 7200].
  --vae-samples <zcount>       Number of samples in VAE z [default: 1]
  --nhidden <nhidden>          Number of hidden dimensions [default: 128].
  --nlatent <nz>               Number of latent VAE dimensions [default: 16].
  --time-print=<sec>           Print status every so often [default: 60].
  --time-sample=<sec>          Print status every so often [default: 600].
  --dump-every=<sec>           Dump model every so often [default: 900]

The data.mat file must contain a (N,d) array of N instances, d dimensions
each.
"""

import time
import yaml
import numpy as np
import h5py
import scipy.io as sio
from docopt import docopt
import pickle

import chainer
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

import model
import util

args = docopt(__doc__, version='train 0.1')
print(args)

print "Using chainer version %s" % chainer.__version__

# Loading training data
# data_mat = h5py.File(args['<data.mat>'], 'r')
# X = data_mat.get('X')
# X = np.array(X)

with open('./moons_tight.pkl', 'rb') as f:
    data = pickle.load(f)
X = data['x']

# X = X.transpose()
N = X.shape[0]
d = X.shape[1]
print "%d instances, %d dimensions" % (N, d)



# Setup model
nhidden = int(args['--nhidden'])
print "%d hidden dimensions" % nhidden
nlatent = int(args['--nlatent'])
print "%d latent VAE dimensions" % nlatent
zcount = int(args['--vae-samples'])
print "Using %d VAE samples per instance" % zcount

vae = model.VAE(d, nhidden, nlatent, zcount)
opt = optimizers.Adam()
opt.setup(vae)
opt.add_hook(chainer.optimizer.GradientClipping(4.0))

# Move to GPU
gpu_id = int(args['--device'])
if gpu_id >= 0:
    cuda.check_cuda_available()
if gpu_id >= 0:
    xp = cuda.cupy
    vae.to_gpu(gpu_id)
else:
    xp = np

# Setup training parameters
batchsize = int(args['--batchsize'])
print "Using a batchsize of %d instances" % batchsize

start_at = time.time()
period_start_at = start_at
period_bi = 0
runtime = int(args['--runtime'])

print_every_s = float(args['--time-print'])
print_at = start_at + print_every_s

sample_every_s = float(args['--time-sample'])
sample_at = start_at + sample_every_s

bi = 0  # batch index
printcount = 0

obj_mean = 0.0
obj_count = 0
counter = 0
with cupy.cuda.Device(gpu_id):
    while True:
        bi += 1
        period_bi += 1

        now = time.time()
        tpassed = now - start_at

        # Check whether we exceeded training time
        if tpassed >= runtime:
            print "Training time of %ds reached, training finished." % runtime
            break

        total = bi * batchsize

        # Print status information
        if now >= print_at:
            print_at = now + print_every_s
            printcount += 1
            tput = float(period_bi * batchsize) / (now - period_start_at)
            EO = obj_mean / obj_count
            print "   %.1fs of %.1fs  [%d] batch %d, E[obj] %.4f,  %.2f S/s, %d total" % \
                  (tpassed, runtime, printcount, bi, EO, tput, total)

            period_start_at = now
            obj_mean = 0.0
            obj_count = 0
            period_bi = 0

        vae.zerograds()

        # Build training batch (random sampling without replacement)
        J = np.sort(np.random.choice(N, batchsize, replace=False))
        x = chainer.Variable(xp.asarray(X[J,:], dtype=np.float32))

        obj = vae(x)
        obj_mean += obj.data
        obj_count += 1

        # (Optionally:) visualize computation graph
        if bi == 1 and args['--vis'] is not None:
            print "Writing computation graph to '%s'." % args['--vis']
            g = computational_graph.build_computational_graph([obj])
            util.print_compute_graph(args['--vis'], g)

        # Update model parameters
        obj.backward()
        opt.update()

        # Sample a set of poses
        if now >= sample_at:
            counter +=1
            sample_at = now + sample_every_s
            print "   # sampling"
            z = np.random.normal(loc=0.0, scale=1.0, size=(16384,nlatent))
            z = chainer.Variable(xp.asarray(z, dtype=np.float32))
            vae.decode(z)
            Xsample = F.gaussian(vae.pmu, vae.pln_var)
            Xsample.to_cpu()
            sio.savemat('samples_%d.mat' % ( counter), { 'X': Xsample.data })
            vae.pmu.to_cpu()
            sio.savemat('means_%d.mat' % (counter), { 'X': vae.pmu.data }) 

# Save model
if args['-o'] is not None:
    modelmeta = args['-o'] + '.meta.yaml'
    print "Writing model metadata to '%s' ..." % modelmeta
    with open(modelmeta, 'w') as outfile:
        outfile.write(yaml.dump(dict(args), default_flow_style=False))

    modelfile = args['-o'] + '.h5'
    print "Writing model to '%s' ..." % modelfile
    serializers.save_hdf5(modelfile, vae)

counter +=1
sample_at = now + sample_every_s
print "   # sampling"
z = np.random.normal(loc=0.0, scale=1.0, size=(16384,nlatent))
z = chainer.Variable(xp.asarray(z, dtype=np.float32))
vae.decode(z)
Xsample = F.gaussian(vae.pmu, vae.pln_var)
Xsample.to_cpu()
sio.savemat('samples_%d.mat' % ( counter), { 'X': Xsample.data })
vae.pmu.to_cpu()
sio.savemat('means_%d.mat' % (counter), { 'X': vae.pmu.data })
