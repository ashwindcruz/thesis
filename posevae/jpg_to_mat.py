#!/usr/bin/env python

import pdb
import scipy
import scipy.io as sio
import numpy as np

# Create training set, 9000 out of the 10,000 that is present at the moment
dim = 218*178*3
training_mat = np.zeros((9000,dim))
for i in range(10000):
	name = str(i) + '.png'
	name.zfill(10)

	image_matrix = scipy.misc.imread(name)
	training_mat[i,:] = np.reshape(image_matrix, (1, dim))
sio.savemat('../celebA_training.mat', {'X': training_mat})