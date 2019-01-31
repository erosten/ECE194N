import numpy as np
import cifar_loader
import matplotlib.pyplot as plt



def split_sets(train_images, train_cls, class_numbers):
	train_set = []
	train_labels = []
	for i in range(class_numbers.shape[0]):
		indices = np.where(train_cls == class_numbers[i])[0]
		train_set.append(train_images[indices])
		train_labels.append(train_cls[indices])

	train_set = np.array(train_set)
	train_set = train_set.reshape([-1,32,32,3])
	train_labels = np.array(train_labels)
	train_labels = train_labels.reshape([-1,])
	return train_set, train_labels

def split_class_names(all_class_names, class_numbers):
	class_names = []
	for i in range(class_numbers.shape[0]):
		class_names.append(all_class_names[class_numbers[i]])

	class_names = np.array(class_names)
	return class_names

def find_examples(train_images, train_cls, labels_to_keep, class_names):
	print(class_names)
	for i in range(labels_to_keep.shape[0]):
		label_class_name = class_names[i]
		rand_integer = np.random.randint(0,4999)
		# grab an example index
		index = np.where(train_cls == labels_to_keep[i])[0][rand_integer]
		img_example = train_images[index]
		imgplot = plt.imshow(img_example)
		plt.title('{} Example'.format(label_class_name))
		plt.show()

def rgb2gray(rgb_imgs):
	gray_imgs = np.zeros((rgb_imgs.shape[0],32,32))
	for i in range(rgb_imgs.shape[0]):
		r, g, b = rgb_imgs[i,:,:,0], rgb_imgs[i,:,:,1], rgb_imgs[i,:,:,2]
		gray_imgs[i,:,:] = (r + g + b) / 3
	return gray_imgs



def compute_dists(train_images, test_images):
	X = test_images
	X_train = train_images
	num_test = X.shape[0]
	num_train = X_train.shape[0]
	dists = np.zeros((num_test, num_train))
	for i in range(num_test):
		print(i)
		for j in range(num_train):
			dists[i,j] = np.linalg.norm(X_train[j,:]-X[i,:])
	return dists

def compute_nearest_neighbors(k, dists):
	nearest_neighbors = np.zeros((dists.shape[0],k))
	print('Finding Indices')
	idx = np.argpartition(dists, k, axis = 0)
	return dists[idx]


train_images, train_cls, train_names = cifar_loader.load_training_data()
test_images, test_cls, test_names = cifar_loader.load_test_data()
class_names = np.array(cifar_loader.load_class_names())


# sort from low to high
labels_to_keep = np.sort(np.array([4, 5, 8, 9]))
train_images, train_cls = split_sets(train_images, train_cls, labels_to_keep)
class_names = split_class_names(class_names, labels_to_keep)

# part a
# find_examples(train_images, train_cls, labels_to_keep, class_names)

# part b

# convert to grayscale
train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)



# dists = compute_dists(train_images, test_images)
# np.save('dists.npy',dists)
# 10000 x 20000
dists = np.load('dists.npy')
print('Saved distances loaded.')
nn = compute_nearest_neighbors(1,dists)
print(nn.shape)
# nn2 = compute_nearest_neighbors(3,dists)
# print(nn2.shape)





