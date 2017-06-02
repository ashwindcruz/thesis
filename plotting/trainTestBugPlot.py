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

# Gather ELBOs for the whole training and testing set
trainingData = []
trainingMeanLowerBound = []
trainingMeanUpperBound = []
with open('plot_issue_train_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		trainingData.append(float(data[0]))
		trainingMeanLowerBound.append(float(data[0]) - 1.96*float(data[1]))
		trainingMeanUpperBound.append(float(data[0]) + 1.96*float(data[1]))
# pdb.set_trace()
testingData = []
testingMeanLowerBound = []
testingMeanUpperBound = []
with open('plot_issue_test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		testingData.append(float(data[0]))
		testingMeanLowerBound.append(float(data[0]) - 1.96*float(data[1]))
		testingMeanUpperBound.append(float(data[0]) + 1.96*float(data[1]))
# pdb.set_trace()
# plt.errorbar(range(len(trainingData)),trainingData, yerr=[trainingMeanLowerBound,trainingMeanUpperBound], ecolor='r', label='train')
#plt.errorbar(range(len(testingData)), testingData, yerr=[testingMeanLowerBound, testingMeanUpperBound], ecolor='g', label='test')
plt.plot(trainingData, label='train')
plt.plot(testingData, label='test')
plt.legend(loc='lower right')
plt.savefig('eval_bug_plot.png')

# Gather ELBOs for the online performance
onlineData = []
onlineMeanLowerBound = []
onlineMeanUpperBound = []
with open('plot_issue_online_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split(',')
		onlineData.append(float(data[0]))
		onlineMeanLowerBound.append(float(data[0]) - 1.96*float(data[1]))
		onlineMeanUpperBound.append(float(data[0]) + 1.96*float(data[1]))
		
# pdb.set_trace()
plt.figure()
# plt.errorbar(range(len(onlineData)), onlineData, yerr=[onlineMeanLowerBound,onlineMeanUpperBound], ecolor='y', label='train')
plt.plot(onlineData)
plt.savefig('online_bug_plot.png')

#ratios = []
#for i in range(0,len(trainingData)):
#    ratios.append(trainingData[i]/testingData[i])

#pdb.set_trace()
