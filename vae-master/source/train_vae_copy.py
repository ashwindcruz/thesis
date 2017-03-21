from chainer import FunctionSet,Variable, datasets, serializers

import chainer.functions as F
import chainer.optimizers as optimizers
import numpy
from vae_mnist import VAE_MNIST_b
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline

vae = VAE_MNIST_b()
optimizer = optimizers.Adam(alpha=0.001)
optimizer.setup(vae)

#from sklearn.datasets import fetch_mldata
#mnist = fetch_mldata('MNIST original')
#x_all = mnist.data.astype(numpy.float32)/255
##y_all = mnist.target.astype(numpy.int32)
#x_train, x_test = numpy.split(x_all, [60000])
##y_train, y_test = numpy.split(y_all, [60000])

# Trying to import the data using the code from the MLP tute instead of above
x_train, x_test = datasets.get_mnist()
#x_train = [element[0] for element in x_train]
#x_test = [element[0] for element in x_test] 
#print(x_train)


#num_epochs = 100
num_epochs = 100

train_free_energies = numpy.zeros(num_epochs)
test_free_energies = numpy.zeros(num_epochs)

batchsize = 100
for epoch in xrange(num_epochs):
    indexes = numpy.random.permutation(60000)
    n_batch = indexes.shape[0]/batchsize
    sum_free_energy = 0

    for i in xrange(0, 60000, batchsize):
        #x_batch = Variable(x_train[indexes[i : i + batchsize]])
        subset = x_train[indexes[i : i + batchsize]][0]
        x_batch = Variable(subset)
        free_energy = vae.free_energy(x_batch)
        sum_free_energy += free_energy.data
        optimizer.zero_grads()
        free_energy.backward()
        optimizer.update()
