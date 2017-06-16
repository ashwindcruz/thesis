#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys

column  = 0
exp_folder_1= './'

# Gather ELBOs for the online performance
online_data_1 = []
with open(exp_folder_1+'/vae_long_5_results/test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_1.append(float(data[column]))
		
# Gather ELBOs for the online performance
online_data_2 = []

with open(exp_folder_1+'/iwae_long_5_results/test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_2.append(float(data[column]))

# Gather ELBOs for the online performance
online_data_3 = []

with open(exp_folder_1+'/householder_long_5_results/test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_3.append(float(data[column]))

max_results = len(online_data_2)


plt.figure()
plt.plot(online_data_1[10:max_results], label='vae')
plt.plot(online_data_2[10:max_results], label='iwae')
plt.plot(online_data_3[10:max_results], label='householder')
plt.legend(loc='lower right')
plt.savefig(exp_folder_1+ '/compare_test_results.png')
