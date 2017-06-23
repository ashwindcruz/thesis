#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys

column  = 0
exp_folder_1= sys.argv[1]
run_number = sys.argv[2]
file = sys.argv[3]

# Gather ELBOs for the online performance
online_data_1 = []
with open(exp_folder_1+'/vae_run_' + sys.argv[2] + '_results/' + file + '_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_1.append(float(data[column]))
		
# Gather ELBOs for the online performance
online_data_2 = []

with open(exp_folder_1+'/iwae_run_' + sys.argv[2] + '_results/' + file + '_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_2.append(float(data[column]))

# Gather ELBOs for the online performance
online_data_3 = []

with open(exp_folder_1+'/householder_run_' + sys.argv[2] + '_results/' + file + '_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_3.append(float(data[column]))

print(len(online_data_1))
print(len(online_data_2))
print(len(online_data_3))

max_results = len(online_data_1)


plt.figure()
plt.plot(online_data_1[10:], label='vae')
plt.plot(online_data_2[10:], label='iwae')
plt.plot(online_data_3[10:], label='householder')
plt.legend(loc='lower right')
plt.savefig(exp_folder_1+ '/compare_' + file + '_' + run_number + '_ELBO.png')
