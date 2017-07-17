#!/usr/bin/env python

import pdb

import sys

import numpy as np
import scipy
import scipy.io as sio

exp_folder = sys.argv[1]
# plt.figure(figsize=(8, 12))
for i in range(int(sys.argv[2])):

    figure_file =  exp_folder + '/samples_' + str(i+1)
    data_file = figure_file + '.mat'
    
    x = sio.loadmat(data_file)['X']
    x = x[0,:]
    # pdb.set_trace()
    x *= 255

    x = np.reshape(x,(64,64,3))

    save_file = figure_file + '.jpg'
    scipy.misc.imsave(save_file,x)

