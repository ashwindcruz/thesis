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
    
    indexes = numpy.random.permutation(10000)   
    sum_test_free_energy = 0
    for i in xrange(0, 10000, batchsize):
        #x_batch = Variable(x_test[indexes[i : i + batchsize]])
	subset = x_test[indexes[i : i + batchsize]][0]
	x_batch = Variable(subset)
        free_energy = vae.free_energy(x_batch)
        sum_test_free_energy += free_energy.data

    train_free_energies[epoch] = sum_free_energy/60000
    test_free_energies[epoch] = sum_test_free_energy/10000
         
    print '[epoch ' +  str(epoch) +']'
    print 'train free energy:' + str(train_free_energies[epoch])
    print 'test free energy:' + str(test_free_energies[epoch])

# Before progressing, first save the model we have. In case there is a bug with the plotting.
serializers.save_npz('firstAttempt.model', vae)

fontsize = 14
plt.plot(numpy.arange(0,num_epochs),train_free_energies,label='train free energy')
plt.plot(numpy.arange(0,num_epochs),test_free_energies,label='test free energy')
plt.legend(fontsize=fontsize)
plt.xlabel('epoch',fontsize=fontsize)
plt.ylabel('$F(X,\\theta,\phi)$',fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
plt.savefig('elbo.png', bbox_inches='tight')


# Reconstructing test data
indexes = numpy.random.permutation(10000)   
#x_batch = Variable(x_test[indexes[:10]])

subset = x_test[indexes[:10]][0]
x_batch = Variable(subset)

x_recon = vae.reconstruct(x_batch)
plt.figure()
for i in xrange(5):
#    plt.subplot(2,5,i)
    plt.subplot(2,5,i+1)
    plt.imshow(x_batch.data[i].reshape(28,28),'binary')
for i in xrange(5):
#    plt.subplot(2,5,i+5)
    plt.subplot(2,5,i+6)
    plt.imshow(x_recon.data[i].reshape(28,28),'binary')
plt.show()
plt.savefig('reconstruct.png', bbox_inches='tight')

# Generating
x_samples = vae.generate(25,False)
plt.figure()
for i in xrange(25):
#    plt.subplot(5,5,i)
    plt.subplot(5,5,i+1)
    plt.imshow(x_samples.data[i].reshape(28,28),'binary')
plt.show()
plt.savefig('generate.png', bbox_inches='tight')

