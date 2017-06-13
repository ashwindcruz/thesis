#!/usr/bin/env python

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import scipy.io

for i in range(int(sys.argv[1])):

	data_file = 'debug_samples_' + str(i+1) + '.mat'
	figure_file = 'image_' + str(i+1)
	x = scipy.io.loadmat(data_file)['X']
	plt.figure()
	plt.scatter(x[:,0], x[:,1], s=1, color='gray')
	plt.savefig(figure_file, bbox_inches='tight')
	plt.close()

