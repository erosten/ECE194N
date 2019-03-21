from __future__ import print_function
import keras
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


#load model
model = load_model('saved_models/keras_cifar10_modified_vgg_dropout.h5')
print('Model Loaded')


# load cifar data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Dataset Loaded')


# preprocess
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# find incorrectly classified images
print('Finding Incorrectly Classified Images')
predictions = model.predict_classes(x_test).reshape((-1,1))
incorrects = np.nonzero(predictions!= y_test)[0]
np.random.shuffle(incorrects)
_, incorrect_indices = np.unique(y_test[incorrects], return_index = True)
incorrect_indices = incorrects[incorrect_indices]
for i in range(incorrect_indices.shape[0]):
	index = incorrect_indices[i]
	imgplt = plt.imshow(x_test[index])
	plt.title('Actual Class: {} Predicted Class: {}'.format(y_test[index], predictions[index]))
	plt.show()



