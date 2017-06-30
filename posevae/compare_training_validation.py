#!/usr/bin/env python

import pdb

import numpy as np
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

exp_folder = sys.argv[1]

def get_results(file, column):
	data_list = []
	with open(file, 'r') as f:
		f.readline()
		for line in f: 
			data = line.split(',')
			data_list.append(float(data[column]))

	return data_list

training_file = exp_folder + '/online_log.txt'
training_results = get_results(training_file,0)
training_results = training_results[0::10]

testing_file = exp_folder + '/test_log.txt'
testing_results = get_results(testing_file,0)

plt.figure()
plt.plot(training_results[30:], label='training')
plt.plot(testing_results[30:], label='testing')
plt.legend(loc='lower right')
plt.savefig(exp_folder+'/training_validation.png', bbox_inches='tight')
