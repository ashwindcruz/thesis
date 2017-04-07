# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:16:27 2017

@author: Ashwin
Plotting code
Compare the training and testing curves for a particular setup
"""

import numpy as np
from matplotlib import pyplot as plt

#plt.xkcd()

elboList1VAE = []

with open('vae/vae01_train_log.txt', 'r') as f:
    f.readline()
    for line in f:
        elbo = float(line)
        elboList1VAE.append(elbo)

elboList2VAE = []

with open('vae/vae02_train_log.txt', 'r') as f:
    f.readline()
    for line in f:
        elbo = float(line)
        elboList2VAE.append(elbo)        

elboList3VAE = []

with open('vae/vae03_train_log.txt', 'r') as f:
    f.readline()
    for line in f:
        elbo = float(line)
        elboList3VAE.append(elbo) 

elboList1IWAE = []

with open('iwae/vae01_train_log.txt', 'r') as f:
    f.readline()
    for line in f:
        elbo = float(line)
        elboList1IWAE.append(elbo)

elboList2IWAE = []

with open('iwae/vae02_train_log.txt', 'r') as f:
    f.readline()
    for line in f:
        elbo = float(line)*70000/(632551/70000)
        elboList2IWAE.append(elbo)        

elboList3IWAE = []

with open('iwae/vae03_train_log.txt', 'r') as f:
    f.readline()
    for line in f:
        elbo = float(line)
        elboList3IWAE.append(elbo) 

        
iterations = list(range(1,len(elboList2IWAE)+1))        
            
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#plt.xticks([])
#plt.yticks([])

#plt.annotate(
#    'WEIRD ELBO MEASURES',
#    xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

plt.plot(iterations, elboList1VAE[0:30], label='k=1 VAE')
plt.plot(iterations, elboList2VAE[0:30], label='k=2 VAE')
plt.plot(iterations, elboList3VAE[0:30], label='k=3 VAE')

plt.plot(iterations, elboList1IWAE[0:30], label='k=1 IWAE')
plt.plot(iterations, elboList2IWAE[0:30], label='k=2 IWAE')
plt.plot(iterations, elboList3IWAE[0:30], label='k=3 IWAE')

plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0.)

plt.savefig('iwae_against_vae.png', bbox_inches='tight')
