import gzip
import numpy as np


image_size = 28
num_train = 60000
num_test = 10000


def load_train_images():

	f = gzip.open('datasets/train-images-idx3-ubyte.gz', 'r')
	f.read(16)

	buff = f.read(image_size * image_size * num_train)
	data = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
	data = data.reshape(num_train, image_size*image_size)

	return data


def load_test_images():

	f = gzip.open('datasets/t10k-images-idx3-ubyte.gz', 'r')
	f.read(16)

	buff = f.read(image_size * image_size * num_test)
	data = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
	data = data.reshape(num_test, image_size*image_size)

	return data

def load_train_labels():

	f = gzip.open('train-labels-idx1-ubyte.gz', 'r')
	f.read(8)

	buff = f.read(num_train)

	label = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
	label = label.reshape(num_train)

	return label

def load_test_labels():

	f = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')
	f.read(8)

	buff = f.read(num_test)

	label = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
	label = label.reshape(num_test)

	return label
