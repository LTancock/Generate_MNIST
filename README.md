With this project I intend to experiment with data synthesis, and seeing how well models perform under varying ratios of natural:artifical data. For this, I will use the MNIST dataset.
The intended approach is to take some number of samples from the original MNIST training data and apply various transformations (e.g. rotation, blur) so that we have the same amount of training data as is normally used in MNIST.
I will then run different models on the original training data and the synthesised data, comparing their accuracies. Sorry to anyone reading this that I haven't made some fancy readme.
