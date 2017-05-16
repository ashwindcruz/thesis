# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:16:27 2017

@author: Ashwin
Plotting code
Compare the training and testing curves for a particular setup
"""

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

trainingData = []
with open('vae1_train_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split()
		trainingData.append(float(data[0]))

testingData = []
with open('vae1_test_log.txt', 'r') as f:
	f.readline()
	for line in f: 
		data = line.split()
		testingData.append(float(data[0]))

plt.plot(trainingData)
plt.plot(testingData)
plt.savefig('bugPlot.png')