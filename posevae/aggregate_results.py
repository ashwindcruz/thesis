#!/usr/bin/env python

import pdb

import numpy as np
import sys

def get_results(file, column):
	data_list = []
	with open(file, 'r') as f:
		f.readline()
		for line in f: 
			data = line.split(',')
			data_list.append(float(data[column]))

	return data_list

def average_results(master_list, min_epoch):
	elbo_collection = np.asarray([elbo[min_epoch-1] for elbo in master_list])
	# print(elbo_collection)
	mean = np.mean(elbo_collection)
	std_dev = np.std(elbo_collection)
	return mean, std_dev


exp_folder = sys.argv[1]
models = ['vae', 'iwae', 'householder']
# models = ['householder']
master_list = []

for model in models: 
	for i in range(3):
		file = exp_folder + 'run_' + str(i+1) + '/' + model + '_run_' + str(i+1) + '_results/test_log.txt'
		data_list = get_results(file, 0)
		master_list.append(data_list)
		print(str(len(data_list)) + ' ' + str(len(data_list)*5))
	
min_epochs = min([len(data_list) for data_list in master_list])
print([len(data_list) for data_list in master_list])

print(str(min_epochs) + ' number of epochs')

mean,std_dev = average_results(master_list[0:3],min_epochs)
print('Vae results -  Mean: ' + str(mean) + ' Std Dev: ' + str(std_dev))

mean,std_dev = average_results(master_list[3:6],min_epochs)
print('IWAE results -  Mean: ' + str(mean) + ' Std Dev: ' + str(std_dev))

mean,std_dev = average_results(master_list[6:9],min_epochs)
print('Householder results -  Mean: ' + str(mean) + ' Std Dev: ' + str(std_dev))

# pdb.set_trace()
