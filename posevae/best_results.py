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

def best_results(master_list_elbo, master_list_sem, min_epoch):
	final_values_elbo = []
	final_values_sem = []
	for indiv_list in master_list_elbo:
		final_values_elbo.append(indiv_list[min_epoch-1])
	for indiv_list in master_list_sem:
		final_values_sem.append(indiv_list[min_epoch-1])
	# pdb.set_trace()
	best_result_elbo = max(final_values_elbo)
	best_index = final_values_elbo.index(best_result_elbo)
	best_result_sem = final_values_sem[best_index]

	return best_result_elbo, best_result_sem


exp_folder = sys.argv[1]
models = ['vae', 'iwae', 'householder_1t', 'householder_10t', 'iaf_1t', 'iaf_2t', 'iaf_3t', 'iaf_4t', 'iaf_8t', 'adgm', 'sdgm']
# models = ['vae', 'iwae']
master_list_elbo = []
master_list_sem = []

for model in models: 
	for i in range(3):
		file = exp_folder + 'run_' + str(i+1) + '/' + model + '_run_' + str(i+1) + '_results/test_log.txt'
		data_list = get_results(file, 0)
		master_list_elbo.append(data_list)
		
		data_list = get_results(file,3)
		master_list_sem.append(data_list)

counter = 0
for model in models:
	elbo, sem = best_results(master_list_elbo[counter:(counter+3)], master_list_sem[counter:(counter+3)], 201)
	print(model + ' results - Mean: ' + str(elbo) + ' SEM: ' + str(sem))
	counter += 3
