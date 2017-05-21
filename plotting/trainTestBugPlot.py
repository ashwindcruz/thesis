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

import pdb

trainingData = []
with open('vae_debug_train_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split()
		trainingData.append(float(data[0]))

testingData = []
with open('vae_debug_test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split()
		testingData.append(float(data[0]))


plt.plot(trainingData, label='train')
plt.plot(testingData, label='test')
plt.legend(loc='lower right')
plt.savefig('bug_plot.png')

#ratios = []
#for i in range(0,len(trainingData)):
#    ratios.append(trainingData[i]/testingData[i])

#pdb.set_trace()
