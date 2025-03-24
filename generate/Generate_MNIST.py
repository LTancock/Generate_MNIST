import numpy as np
from os.path  import join
from Read_MNIST import MnistDataloader
import matplotlib.pyplot as plt


input_path = 'input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#ABOVE IS NOT MY CODE, copied from Read_MNIST

#number of samples to pull from training set
noSamples = 1
# Create a 1D NumPy array of empty lists, treating each element as an object
samples = np.empty(10, dtype=object)

# Initialize each element as an empty list
for i in range(10):
    samples[i] = []


#Don't know the ordering of MNIST data so just searching through linearly for each thing, might change this later
#y_train[x] is the number presented in the image in position x
for i in range(0, len(x_train)):
    #print(y_train[i])
    if len(samples[y_train[i]]) < noSamples:
        samples[y_train[i]].append(x_train[i])
        


plt.figure(figsize=(30,20))
for x in samples:
    #print(x[0])
    plt.imshow(x[0])
    plt.show()