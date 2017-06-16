#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys

column  = 5

# Gather ELBOs for the online performance
online_data_1 = []
exp_folder_1= './long_3'
with open(exp_folder_1+'/vae_long_3_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_1.append(float(data[column]))
		
# Gather ELBOs for the online performance
online_data_2 = []

with open(exp_folder_1+'/iwae_long_3_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_2.append(float(data[column]))

# Gather ELBOs for the online performance
online_data_3 = []

with open(exp_folder_1+'/householder_long_3_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_3.append(float(data[column]))

max_results = len(online_data_3)

online_data_4 = []
exp_folder_2= './long_4'
with open(exp_folder_2+'/vae_long_4_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_4.append(float(data[column]))
		
# Gather ELBOs for the online performance
online_data_5 = []

with open(exp_folder_2+'/iwae_long_4_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_5.append(float(data[column]))

# Gather ELBOs for the online performance
online_data_6 = []

with open(exp_folder_2+'/householder_long_4_results/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data_6.append(float(data[column]))

max_results = min(len(online_data_3),len(online_data_6))



plt.figure()
plt.plot(online_data_1[0:max_results], label='vae_3')
plt.plot(online_data_4[0:max_results], label='vae_4')
# plt.ylim([0,0.01])
plt.legend(loc='lower right')
plt.savefig(exp_folder_1+ '/compare_vae_results.png')

plt.figure()
plt.plot(online_data_2[0:max_results], label='iwae_3')
plt.plot(online_data_5[0:max_results], label='iwae_4')
# plt.ylim([0,0.01])
plt.legend(loc='lower right')
plt.savefig(exp_folder_1+ '/compare_iwae_results.png')


plt.figure()
plt.plot(online_data_3[0:max_results], label='householder_3')
plt.plot(online_data_6[0:max_results], label='householder_4')
# plt.ylim([0,0.01])
plt.legend(loc='lower right')
plt.savefig(exp_folder_1+ '/compare_householder_results.png')
