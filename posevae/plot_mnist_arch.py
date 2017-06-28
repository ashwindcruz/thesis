#!/usr/bin/env python
"""
Created on Tue May 16 21:16:27 2017

@author: Ashwin
Plotting code
Compare the training and testing curves for a particular setup
"""

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys

import pdb

exp_folder = sys.argv[1]

settings = ['128_32','128_64','256_32','256_64','512_32','512_64']

# Gather ELBOs for the whole training and testing set
def parse_results(file):
	training_data = []
	with open(file, 'r') as f:
		f.readline()
		for line in f: 
			data = line.split(',')
			training_data.append(float(data[0]))
	return training_data


for i in range(3):
	master_list = []
	for setting in settings:
		file = exp_folder + '/vae_' + setting + '_run_' + str(i+1) + '_results/test_log.txt'
		results = parse_results(file) 
		master_list.append(results)

	plt.figure()
	for j in range(6):
		plt.plot(master_list[j][10:], label=settings[j])
	plt.legend(loc='lower right')
	plt.savefig(exp_folder+'/run_' + str(i+1)+'.png')
	plt.close()