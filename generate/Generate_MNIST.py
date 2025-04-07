import numpy as np
from os.path  import join
import scipy.ndimage
from Read_MNIST import MnistDataloader
import matplotlib.pyplot as plt
import scipy
import random


def add_gaussian_noise(sample, mean=0, std=10):
    noise = np.random.normal(loc=mean, scale=std, size=(len(sample), len(sample[0])))
    noisyImage = sample + noise
    # Clip values to keep them in valid range [0, 255]
    noisyImage = np.clip(noisyImage, 0, 255)
    return noisyImage

#Transformations
#Focussing on transformations that are not easily corrected for, as it isn't as difficult to fix those in the testing stage

#Rotation
def rotateMNIST(sample, noRotations, angle):
    rotated = []
    for x in range (0, noRotations):
        rotated.append(scipy.ndimage.rotate(sample, angle=random.randint(-angle, angle), reshape=False, mode='constant', cval=0.0))
    return rotated

#Gaussian blur
def blurMNIST(sample, noBlurs):
    blurred = []
    for x in range(0, noBlurs):
        blurred.append(scipy.ndimage.gaussian_filter(sample, sigma=random.uniform(0,1)))
    return blurred

#Gaussian noise
def noisyMNIST(sample, noNoise, deviation):
    noised = []
    for x in range(0, noNoise):
        noised.append(add_gaussian_noise(sample, std=deviation))
    return noised


input_path = 'input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#ABOVE IS NOT MY CODE, copied from Read_MNIST

#number of samples to pull of each digit from training set
noSamples = 1
# Create a 1D NumPy array of empty lists, treating each element as an object
samples = np.empty(10, dtype=object)

# Initialize each element as an empty list, creating one for each digit
for i in range(10):
    samples[i] = []

#Don't know the ordering of MNIST data so just searching through linearly for each thing, might change this later
#y_train[x] is the number presented in the image in position x
for i in range(0, len(x_train)):
    #print(y_train[i])
    if len(samples[y_train[i]]) < noSamples:
        samples[y_train[i]].append(x_train[i])
        


#Showing the samples
#plt.figure(figsize=(30,20))
#for x in samples:
    #plt.imshow(x[0])
    #plt.show()
plt.imshow(samples[0][0])
plt.show()
#print(samples[0])
#print("\n\n\n\n\n\n")
#print(rotateMNIST(samples[0][0]))

#Rotations
#rotations = rotateMNIST(samples[0][0], 5, 15)
#for x in rotations:
#    plt.imshow(x)
#    plt.show()

#Gaussian Blur
#Effect is subdued on low pixel density images, not sure that it will do much useful/might cause overfitting
#blurred = blurMNIST(samples[0][0], 5)
#for x in blurred:
#    plt.imshow(x)
#    plt.show()

#Gaussian Noise
#Not sure how much to put on the deviation, will try out a few but probably quite small
#20 is the around the amount you'd expect, 50 is relatively bad quality
noised = noisyMNIST(samples[0][0], 5, 20)
for x in noised:
    plt.imshow(x)
    plt.show()