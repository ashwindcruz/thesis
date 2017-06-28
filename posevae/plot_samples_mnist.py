#!/usr/bin/env python

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import scipy.io

exp_folder = sys.argv[1]
# plt.figure(figsize=(8, 12))
for i in range(int(sys.argv[2])):

    data_file = exp_folder + '/samples_' + str(i+1) + '.mat'
    figure_file =  exp_folder + '/samples_' + str(i+1)
    x = scipy.io.loadmat(data_file)['X']

# plt.subplot(5, 2, 2*i + 1)
    plt.figure()
    plt.imshow(x[1].reshape(28,28),vmin=0, vmax=1, cmap="gray")
# plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.savefig(figure_file, bbox='tight')
    plt.tight_layout()
    plt.close()

