#!/usr/bin/env python 

import pdb

import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kinect_visualize_v1
import numpy as np
import h5py

# Samples from our model
#M = scipy.io.loadmat('vae01_samples.mat')
#X = M['X']

# Samples from real data
X = h5py.File('../posevae/data/MSRC12-X-d60.mat', 'r')
X = X.get('X')
X = np.array(X)
X = X.T

#for i in range((X.shape[0])):
for i in range(10):
	X_i = np.asarray(X[i,:])
	#pdb.set_trace()
	fig,axes = kinect_visualize_v1.visualize_skeleton(X_i, fsize=6)
	fig.savefig('vae01_samples_%d.png' %i)

#pdb.set_trace()