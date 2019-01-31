import pickle
import numpy as np

# reference: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py


img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10

num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert_raw_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def load_data(filename):
    data = unpickle(filename)
    raw_images = data[b'data']
    class_numbers = np.array(data[b'labels'])
    images = convert_raw_images(raw_images)
    return images, class_numbers

def load_class_names():
    raw = unpickle(file = 'cifar-10-batches-py/batches.meta')[b'label_names']
    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]
    return names

def load_training_data():

	# Pre-allocate the arrays for the images and class-numbers for efficiency.
	images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
	cls = np.zeros(shape=[num_images_train], dtype=int)

	# Begin-index for the current batch.
	begin = 0

	# For each data-file.
	for i in range(num_files_train):
	    # Load the images and class-numbers from the data-file.
	    images_batch, cls_batch = load_data(filename='cifar-10-batches-py/data_batch_' + str(i + 1))

	    # Number of images in this batch.
	    num_images = len(images_batch)

	    # End-index for the current batch.
	    end = begin + num_images

	    # Store the images into the array.
	    images[begin:end, :] = images_batch

	    # Store the class-numbers into the array.
	    cls[begin:end] = cls_batch

	    # The begin-index for the next batch is the current end-index.
	    begin = end

	return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = load_data(filename='cifar-10-batches-py/test_batch')

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

