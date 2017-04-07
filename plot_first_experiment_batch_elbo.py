# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:16:27 2017

@author: Ashwin
Plotting code
Plot the ELBO per batch based on the log files with the format provided in 'First Experiment'
"""

import numpy as np
from matplotlib import pyplot as plt

#plt.xkcd()

elboList = []

with open('vae_01_log.txt', 'r') as f:
    for line in f:
        words = line.split()
        wordCount = len(words)
        if(wordCount==12):
            elbo = float(words[7][0:-1])
            elboList.append(elbo)
            
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#plt.xticks([])
#plt.yticks([])

#plt.annotate(
#    'WEIRD ELBO MEASURES',
#    xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

plt.plot(elboList)

plt.xlabel('Printing Iterations')
plt.ylabel('Negative ELBO')

