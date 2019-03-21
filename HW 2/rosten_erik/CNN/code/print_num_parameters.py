from __future__ import print_function
import keras
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


model = load_model('saved_models/keras_cifar10_modified_vgg_dropout.h5')
model.summary()



