#!/usr/bin/env python

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import scipy.io

exp_folder = sys.argv[1]
for i in range(int(sys.argv[2])):

	data_file = exp_folder + '/samples_' + str(i+1) + '.mat'
	figure_file = exp_folder + '/samples_' + str(i+1)
	x = scipy.io.loadmat(data_file)['X']
	plt.figure()
	plt.scatter(x[0:2048,0], x[0:2048,1], s=1, color='gray')
	plt.savefig(figure_file, bbox_inches='tight')
	plt.close()

        data_file = exp_folder + '/means_' + str(i+1) + '.mat'
	figure_file = exp_folder + '/means' + str(i+1)
	x = scipy.io.loadmat(data_file)['X']
	plt.figure()
	plt.scatter(x[:,0], x[:,1], s=1, color='gray')
	plt.savefig(figure_file, bbox_inches='tight')
	plt.close()

