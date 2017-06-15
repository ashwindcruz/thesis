#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys

# Gather ELBOs for the online performance
online_data_1 = []
exp_folder= '.'
with open(exp_folder+'/vae_overnight_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_1.append(float(data[0]))
		
# Gather ELBOs for the online performance
online_data_2 = []

with open(exp_folder+'/iwae_overnight_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_2.append(float(data[0]))

# Gather ELBOs for the online performance
online_data_3 = []

with open(exp_folder+'/householder_overnight_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_3.append(float(data[0]))

max_results = len(online_data_3)

plt.figure()
plt.plot(online_data_1[0:max_results], label='vae')
plt.plot(online_data_2[0:max_results], label='iwae')
plt.plot(online_data_3[0:max_results], label='householder')
plt.ylim([0,200])
plt.legend(loc='lower right')
plt.savefig(exp_folder+ '/compare_online_results.png')
