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

hidden = [256, 512]
layers = [2,3]
latent = [128,256]

# Gather ELBOs for the whole training and testing set
def parse_results(file):
	training_data = []
	with open(file, 'r') as f:
		f.readline()
		for line in f: 
			data = line.split(',')
			training_data.append(float(data[0]))
	return training_data

master_list = []
legends = []
for units in hidden:
    for layer in layers:
        for dims in latent:

	    file = exp_folder + '/vae_hidden_' + str(units) + '_layers_' + str(layer) + '_latent_' + str(dims) + '_results/online_log.txt'
	    results = parse_results(file) 
	    master_list.append(results)
            legends.append(file[14:-17])

plt.figure()
for j in range(8):
    plt.plot(master_list[j][10:], label=legends[j])
    plt.legend(loc='lower right')
plt.savefig(exp_folder+'/celeb_training_arch.png')
plt.close()
