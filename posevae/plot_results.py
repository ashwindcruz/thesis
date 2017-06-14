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

# Gather ELBOs for the whole training and testing set
training_data = []
trainingMeanLowerBound = []
trainingMeanUpperBound = []
with open(exp_folder+'/train_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		training_data.append(float(data[0]))
		trainingMeanLowerBound.append(float(data[0]) - 1.96*float(data[1]))
		trainingMeanUpperBound.append(float(data[0]) + 1.96*float(data[1]))

testing_data = []
testingMeanLowerBound = []
testingMeanUpperBound = []
difference = []
counter = 0
with open(exp_folder+'/test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		testing_data.append(float(data[0]))
		testingMeanLowerBound.append(float(data[0]) - 1.96*float(data[1]))
		testingMeanUpperBound.append(float(data[0]) + 1.96*float(data[1]))

		difference.append(training_data[counter]-float(data[0]))
		counter += 1

plt.plot(training_data, label='train')
plt.plot(testing_data, label='test')
plt.legend(loc='lower right')
plt.savefig('eval_bug_plot.png')

plt.figure()
plt.plot(difference)
plt.savefig('difference.png')

# Gather ELBOs for the online performance
online_data = []
onlineMeanLowerBound = []
onlineMeanUpperBound = []
with open(exp_folder+'/online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		online_data.append(float(data[0]))
		onlineMeanLowerBound.append(float(data[0]) - 1.96*float(data[1]))
		onlineMeanUpperBound.append(float(data[0]) + 1.96*float(data[1]))
		

plt.figure()
plt.plot(online_data)
plt.savefig('online_bug_plot.png')

