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

def compute_nearest_neighbors(k, dists, train_cls):
	nearest_neighbors = np.zeros((dists.shape[0],k))
	nearest_neighbor_classes = np.zeros((dists.shape[0],k))
	for i in range(dists.shape[0]):
		idx = np.argpartition(dists[i,:], k, axis = 0)[:k]
		nearest_neighbors[i,:] = dists[i,idx]
		nearest_neighbor_classes[i,:] = train_cls[idx]
	return nearest_neighbors, nearest_neighbor_classes

def compute_error_rate(nn, test_cls , nn_cls):
	acc_count = 0
	total_test_images = nn.shape[0]
	k = nn.shape[1]
	nn.astype(int)
	nn_cls.astype(int)
	# if k is even
	# if (k % 2 == 0 ):
	for i in range(total_test_images):
		unique_labels = np.unique(nn_cls[i])
		label_count = 0
		label = 5
		for j in range(unique_labels.shape[0]):
			num_occurences = np.count_nonzero(nn_cls[i] == unique_labels[j])
			if (num_occurences > label_count):
				label_count = num_occurences
				label = unique_labels[j]
			elif(num_occurences == label_count):
				# print('Unique Labels: {}'.format(unique_labels))
				# print('Num Occurences of {} is {}'.format(unique_labels[j], num_occurences))
				ind1 = np.where(nn_cls[i] == unique_labels[j])
				ind2 = np.where(nn_cls[i] == label)
				avg1 = np.average(nn[i,ind1])
				avg2 = np.average(nn[i,ind2])
				# print(ind1)
				# print(ind2)
				# print(avg1)
				# print(avg2)
				if (avg1 < avg2):
					label = unique_labels[j]
				# print(label)
		if (test_cls[i] != np.int(label)):
			acc_count = acc_count + 1

	return acc_count / total_test_images	


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
k_array = np.array([1, 2, 4, 5, 9, 10, 19, 20])
for k in range(k_array.shape[0]):
	nn, nn_cls = compute_nearest_neighbors(k_array[k], dists, train_cls)
	e_rate = compute_error_rate(nn, test_cls, nn_cls)
	print('Error rate for k = {} is {}'.format(k_array[k], e_rate))
# nn2 = compute_nearest_neighbors(3,dists)
# print(nn2.shape)





