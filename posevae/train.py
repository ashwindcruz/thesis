#!/usr/bin/env python

"""Train variational autoencoder (VAE) model for pose data.

Author: Sebastian Nowozin <senowozi@microsoft.com>, Ashwin D'Cruz <ashwindcruz94@gmail.com>
Date: 5th August 2016

Usage:
  train.py (-h | --help)
  train.py [options] <data.mat>

Options:
  -h --help                     Show this help screen.
  -g <device>, --device         GPU id to train model on.  Use -1 for CPU [default: -1].
  --model-type <model-type>     Type of model to use (examples: vae, iwae, householder, planar) [default: vae].
  -o <model-prefix>             Write trained model to given file.h5 [default: output].
  --vis <graph.ext>             Visualize computation graph.
  -b <batchsize>, --batchsize   Minibatch size [default: 100].
  --batch-limit <batch-limit>   Total number of batches to process for training [default: -1]
  -t <runtime>, --runtime       Total training runtime in seconds [default: 7200].
  --nhidden <nhidden>           Number of hidden dimensions [default: 256].
  --nlatent <nz>                Number of latent VAE dimensions [default: 16].
  --time-print=<sec>            Print status every so often [default: 60].
  --time-sample=<sec>           Print status every so often [default: 600].
  --dump-every=<sec>            Dump model every so often [default: 900].
  --log-interval <log-interval> Number of batches before logging training and testing ELBO [default: 100].
  --test <test>                 Number of samples to set aside for testing [default:70000].
  --vae-samples <zcount>        Number of samples in VAE/IWAE z [default: 1].
  --house-degree <house-degree> Number of Householder flow transformations to apply. Only applicable with householder model. [default: 1]. 
  --nmap <nmap>                 Number of Planar flow mappings to apply. Only applicable with planar model. [default:1].  

The data.mat file must contain a (N,d) array of N instances, d dimensions
each.
"""


import time
import yaml
import numpy as np
import h5py
import scipy.io as sio
from docopt import docopt
import os

import chainer
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

from vae_SEM import model as vae 
from iwae import model as iwae 
from householder import model as householder
from planar import model as planar 

import util


np.random.seed(0) # For debugging purposes, this removes one factor of influece from our experiments

import pdb

args = docopt(__doc__, version='train 0.1')
print(args)

print "Using chainer version %s" % chainer.__version__

# Loading training data
data_mat = h5py.File(args['<data.mat>'], 'r')
X = data_mat.get('X')
X = np.array(X)
X = X.transpose()
N = X.shape[0]
d = X.shape[1]
print "%d instances, %d dimensions" % (N, d)

# Split data into training and testing data
# In case the there is a sequential relationship in the data. The seed fixing at the start ensures that the shuffle is consistent between experimental runs.
X  = np.random.permutation(X)
test_size = int(args['--test'])
X_test = X[0:test_size,:]
X_train = X[test_size:,:]

N = X_train.shape[0]
N_test = X_test.shape[0]
#N -= test_size

# Set up model
nhidden = int(args['--nhidden'])
print "%d hidden dimensions" % nhidden
nlatent = int(args['--nlatent'])
print "%d latent VAE dimensions" % nlatent
zcount = int(args['--vae-samples'])
print "Using %d VAE samples per instance" % zcount

log_interval = int(args['--log-interval'])
print "Recording training and testing ELBO every %d batches" % log_interval

# Check which model was specified
model_type = args['--model-type']
if model_type=='vae':
    vae = vae.VAE(d, nhidden, nlatent, zcount)
elif model_type=='iwae':
    vae = iwae.VAE(d, nhidden, nlatent, zcount)
elif model_type=='householder':
    hdegree = int(args['--house-degree'])
    print 'Using %d Householder flow transformations' % hdegree
    vae = householder.VAE(d, nhidden, nlatent, zcount, hdegree)
elif model_type=='planar':
    nmap = int(args['--nmap'])
    print 'Using %d Planar flow mappings' % nmap
    vae = planar.VAE(d, nhidden, nlatent, zcount, nmap)


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
batch_size = int(args['--batchsize'])
print "Using a batchsize of %d instances" % batch_size
batch_limit = int(args['--batch-limit'])
if batch_limit!=-1:
    print "Limiting training to run for %d batches" % batch_limit

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

<<<<<<< HEAD
=======
counter = 0
pdb.set_trace()
>>>>>>> 3d1965b... Fixing repo
# Folder where results will be saved
directory = model_type + '_' + args['-o'] + '_results'
if not os.path.exists(directory):
    os.makedirs(directory)


with cupy.cuda.Device(gpu_id):
    # Set up variables that cover the entire training and testing sets
    # x_train = chainer.Variable(xp.asarray(X_train, dtype=np.float32))
    # x_test = chainer.Variable(xp.asarray(X_test, dtype=np.float32))
    
    # Set up the training and testing log files
    online_log_file = directory + '/' + args['-o'] + '_online_log.txt' 
    train_log_file = directory + '/' + args['-o'] + '_train_log.txt'
    test_log_file  = directory + '/' + args['-o'] +  '_test_log.txt' 

    with open(online_log_file, 'w+') as f:
        f.write('Online Log \n')

    with open(train_log_file, 'w+') as f:
        f.write('Training Log \n')

    with open(test_log_file, 'w+') as f:
        f.write('Testing Log \n')
    
    while True:
        bi += 1
        period_bi += 1

        now = time.time()
        tpassed = now - start_at

        # Check whether we exceeded training time
        if tpassed >= runtime:
            print "Training time of %ds reached, training finished." % runtime
            break

        # Check whether we exceeded the batch limit
        if bi > batch_limit:
            print " Batch limit of %d reached, training finished." % batch_limit
            break

        total = bi * batch_size

        # Print status information
        if now >= print_at:
        #if True:
            print_at = now + print_every_s
            printcount += 1
            tput = float(period_bi * batch_size) / (now - period_start_at)
            if(obj_count==0):
                obj_count+=1
            EO = obj_mean / obj_count
            #pdb.set_trace()
            print "   %.1fs of %.1fs  [%d] batch %d, E[obj] %.4f,  %.2f S/s, %d total" % \
                  (tpassed, runtime, printcount, bi, EO, tput, total)

            period_start_at = now
            obj_mean = 0.0
            obj_count = 0
            period_bi = 0

<<<<<<< HEAD
        vae.zerograds()

        # Build training batch (random sampling without replacement)
        J = np.sort(np.random.choice(N, batch_size, replace=False))
        x = chainer.Variable(xp.asarray(X_train[J,:], dtype=np.float32))
        
        obj, timing_info = vae(x)

        obj_mean_new = float((F.sum(obj)/batch_size).data)
        obj_mean_new_sem = obj.data.std()/xp.sqrt(batch_size)
        

        # Update model parameters, if the average objective is valid
        # if(not np.isinf(obj_mean_new)):
        obj_mean += obj_mean_new 
        obj_count += 1
        obj_mean_variable = F.sum(obj)/batch_size # For some reason F.mean is not being recognized. Perhaps it is not included in this particular version of Chainer. 
        backward_timing = time.time()
        obj_mean_variable.backward()
        opt.update()
        backward_timing = time.time() - backward_timing
        # pdb.set_trace()
        # Log the information 
        with open(online_log_file, 'a') as f:
                f.write(str(obj_mean_new) + ',' + str(obj_mean_new_sem) + ',' + str(timing_info[0]) + ',' + str(timing_info[1]) + ',' + str(backward_timing) + '\n')


            # EO = obj_mean / obj_count
            
            #pdb.set_trace()
            # print "   %.1fs of %.1fs batch %d, E[obj] %.4f,  %d total" % \
                  # (tpassed, runtime, bi, EO, total)
        # else:
        #     print('Unused batch due to occurence of inf.\n')
=======
        X_online = np.random.permutation(X_train)
        vae, obj_mean = util.evaluate_dataset(vae, X_online, batch_size, online_log_file, True, opt)
      
        # If the model breaks, terminate training early
        if(math.isnan(vae.obj.data)):
            if args['-o'] is not None:
                modelmeta = directory + '/' + 'pre_break_meta.yaml'
                print "Writing model metadata to '%s' ..." % (modelmeta)
                with open(modelmeta, 'w') as outfile:
                    outfile.write(yaml.dump(dict(args), default_flow_style=False))
            break
>>>>>>> 3d1965b... Fixing repo

        # Get the ELBO for the training and testing set and record it
        # -1 is because we want to record the first set which has bi value of 1
        if((bi-1)%log_interval==0):
            eval_batch_size = 8192
            
            print('##################### Beginning Training Evaluation #####################')
            # Training results
            training_ELBO = xp.zeros([N])
            training_timing_info = np.array([0.,0.])
            for i in range(0,N/eval_batch_size):
                x_train_c = chainer.Variable(xp.asarray(X_train[i*eval_batch_size:(i+1)*eval_batch_size,:], dtype=np.float32), volatile='ON')
                obj_train, obj_train_timing = vae(x_train_c)
        
                training_ELBO[i*eval_batch_size:(i+1)*eval_batch_size] = obj_train.data
                training_timing_info += obj_train_timing

            # One final smaller batch to cover what couldn't be captured in the loop
            x_train_c = chainer.Variable(xp.asarray(X_train[(N/eval_batch_size)*eval_batch_size:,:], dtype=np.float32), volatile='ON')
            obj_train, obj_train_timing = vae(x_train_c)
            # pdb.set_trace()    
            training_ELBO[(N/eval_batch_size)*eval_batch_size:] = obj_train.data 
            # Don't use the latest information because the batchsize could be wildly different.
            # Also, the division will yield the rounded down integer which is why we don't have a -1
            training_timing_info /= (N/eval_batch_size) 

            # Calculate the average and the SEM of the training ELBOs
            training_obj = -training_ELBO.mean()
            training_std = training_ELBO.std()
            training_sem = training_std/xp.sqrt(N)

            with open(train_log_file, 'a') as f:
                f.write(str(training_obj) + ',' + str(training_sem) + ',' + str(training_timing_info[0]) + ',' + str(training_timing_info[1]) + '\n')
            
            print('##################### Beginning Testing Evaluation   #####################')
            # Testing results
            testing_ELBO = xp.zeros([N_test])
            testing_timing_info = np.array([0.,0.])
            for i in range(0,N_test/eval_batch_size):
                x_test_c = chainer.Variable(xp.asarray(X_test[i*eval_batch_size:(i+1)*eval_batch_size,:], dtype=np.float32), volatile='ON')
                obj_test, obj_test_timing = vae(x_test_c)
        
                testing_ELBO[i*eval_batch_size:(i+1)*eval_batch_size] = obj_test.data
                testing_timing_info += obj_test_timing

            # One final smaller batch to cover what couldn't be captured in the loop
            x_test_c = chainer.Variable(xp.asarray(X_test[(N_test/eval_batch_size)*eval_batch_size:,:], dtype=np.float32), volatile='ON')
            obj_test, obj_test_timing = vae(x_test_c)
            # pdb.set_trace()
            testing_ELBO[(N_test/eval_batch_size)*eval_batch_size:] = obj_test.data 
            # Don't use the latest information because the batchsize could be wildly different.
            # Also, the division will yield the rounded down integer which is why we don't have a -1
            testing_timing_info /= (N_test/eval_batch_size) 

            # Calculate the average and the SEM of the training ELBOs
            testing_obj = -testing_ELBO.mean()
            testing_std = testing_ELBO.std()
            testing_sem = testing_std/xp.sqrt(N)

            with open(test_log_file, 'a') as f:
                f.write(str(testing_obj) + ',' + str(testing_sem) + ',' + str(testing_timing_info[0]) + ',' + str(testing_timing_info[1]) + '\n')

            print('##################### Saving Model Checkpoint     #####################')
            # Save model
<<<<<<< HEAD
            if args['-o'] is not None:
=======
            if ((args['-o'] is not None) and ((bi-1)%(log_interval*100)==0)): #Additional *5 term because we don't want a checkpoint every log point
                #print('##################### Saving Model Checkpoint     #####################')
>>>>>>> 3d1965b... Fixing repo
                batch_number = str(bi).zfill(6)
                modelfile = directory + '/' + args['-o'] + '_' + batch_number + '.h5'
                print "Writing model checkpoint to '%s' ..." % (modelfile)
                serializers.save_hdf5(modelfile, vae)

        # (Optionally:) visualize computation graph
        if bi == 1 and args['--vis'] is not None:
            print "Writing computation graph to '%s/%s'." % (directory,args['--vis'])
            g = computational_graph.build_computational_graph([obj])
            util.print_compute_graph(directory + '/' + args['--vis'], g)

        # Sample a set of poses
        if now >= sample_at:
            sample_at = now + sample_every_s
            print "   # sampling"
            z = np.random.normal(loc=0.0, scale=1.0, size=(1024,nlatent))
            z = chainer.Variable(xp.asarray(z, dtype=np.float32))
            vae.decode(z)
            Xsample = F.gaussian(vae.pmu, vae.pln_var)
            Xsample.to_cpu()
            sio.savemat('%s/%s_samples_%d.mat' % (directory,args['-o'], total), { 'X': Xsample.data })


# Save model
if args['-o'] is not None:
    modelmeta = directory + '/' + args['-o'] + '.meta.yaml'
    print "Writing model metadata to '%s' ..." % (modelmeta)
    with open(modelmeta, 'w') as outfile:
        outfile.write(yaml.dump(dict(args), default_flow_style=False))

    modelfile = directory + '/' + args['-o'] + '.h5'
    print "Writing final model to '%s' ..." % (modelfile)
    serializers.save_hdf5(modelfile, vae)

