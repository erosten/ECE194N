import numpy as np
import cifar_loader
import matplotlib.pyplot as plt
import os



def split_sets(all_train_images, all_train_classes, labels_to_keep):
	images = []
	classes = []
	for i in range(labels_to_keep.shape[0]):
		indices = np.where(all_train_classes == labels_to_keep[i])[0]
		images.append(all_train_images[indices])
		classes.append(all_train_classes[indices])

	images = np.array(images)
	images = images.reshape([-1,32,32,3])
	classes = np.array(classes)
	classes = classes.reshape([-1,])
	return images, classes

def split_class_names(all_class_names, class_numbers):
	class_names = []
	for i in range(class_numbers.shape[0]):
		class_names.append(all_class_names[class_numbers[i]])
	return np.array(class_names)

def view_examples(train_imgs, train_cls, labels_to_keep, class_names):
	print(class_names)
	for i in range(labels_to_keep.shape[0]):
		label_class_name = class_names[i]
		rand_integer = np.random.randint(0,4999)
		# grab a random example index
		index = np.where(train_cls == labels_to_keep[i])[0][rand_integer]
		img_example = train_imgs[index]
		imgplot = plt.imshow(img_example)
		plt.title('{} Example'.format(label_class_name))
		plt.show()


def rgb2gray_average(rgb_imgs):
	gray_imgs = np.zeros((rgb_imgs.shape[0],32,32))
	for i in range(rgb_imgs.shape[0]):
		r, g, b = rgb_imgs[i,:,:,0], rgb_imgs[i,:,:,1], rgb_imgs[i,:,:,2]
		gray_imgs[i,:,:] = (r + g + b) / 3
	return gray_imgs

def rgb2gray_luminosity(rgb_imgs):
	gray_imgs = np.zeros((rgb_imgs.shape[0],32,32))
	for i in range(rgb_imgs.shape[0]):
		r, g, b = rgb_imgs[i,:,:,0], rgb_imgs[i,:,:,1], rgb_imgs[i,:,:,2]
		gray_imgs[i,:,:] = 0.21 * r + 0.72 * g + 0.07 * b
	return gray_imgs

def euclidean_dist(img1, img2):
	return np.linalg.norm(img1-img2)

def ssd(img1, img2):
	return np.sum((img1-img2)**2)

def compute_dists(train_imgs, test_imgs):
	num_test = test_imgs.shape[0]
	num_train = train_imgs.shape[0]
	dists = np.zeros((num_test, num_train))
	for i in range(num_test):
		for j in range(num_train):
			dists[i,j] = ssd(train_imgs[j,:], test_imgs[i,:])
	return dists

def find_nearest_neighbors(k, dists, train_cls):
	nearest_neighbors = np.zeros((dists.shape[0],k))
	nearest_neighbor_classes = np.zeros((dists.shape[0],k))
	nearest_neighbor_class_indices = np.zeros((dists.shape[0],k))
	for i in range(dists.shape[0]):
		idx = np.argpartition(dists[i,:], k, axis = 0)[:k]
		nearest_neighbors[i,:] = dists[i,idx]
		nearest_neighbor_classes[i,:] = train_cls[idx]
		nearest_neighbor_class_indices[i,:] = idx
	return nearest_neighbors, nearest_neighbor_classes.astype(int), nearest_neighbor_class_indices.astype(int)

def compute_error_rate(nn, test_cls , nn_cls):
	acc_count = 0
	total_test_imgs = nn.shape[0]
	k = nn.shape[1]

	for i in range(total_test_imgs):
		unique_labels = np.unique(nn_cls[i])
		label_count = 0
		label = np.max(unique_labels) + 1
		for j in range(unique_labels.shape[0]):
			num_occurences = np.count_nonzero(nn_cls[i] == unique_labels[j])
			if (num_occurences > label_count):
				label_count = num_occurences
				label = unique_labels[j]
			elif(num_occurences == label_count):
				new_indices = np.where(nn_cls[i] == unique_labels[j])[0]
				old_indices = np.where(nn_cls[i] == label)[0]
				new_avg = np.average(nn[i,new_indices])
				old_avg = np.average(nn[i,old_indices])
				if (new_avg < old_avg):
					label = unique_labels[j]

		if (test_cls[i] != np.int(label)):
			acc_count = acc_count + 1

	return acc_count / total_test_imgs	

def plot_nearest_neighbors(labels_to_keep, dists, train_cls, test_cls, test_imgs, train_imgs, class_names):

	for i in range(labels_to_keep.shape[0]):
		label = labels_to_keep[i]
		nn, nn_cls, nn_cls_indices = find_nearest_neighbors(5, dists, train_cls)
		label_indices = np.where(test_cls == label)[0]
		ind = np.random.randint(0, label_indices.shape[0] - 1)
		label_index = label_indices[ind]
		nn_img_indices = nn_cls_indices[label_index,:]
		nn_cls = nn_cls[label_index]
		nn = nn[label_index]

		plt.figure(1)
		ax1 = plt.subplot(2,3,1)
		ax1.axis('off')
		imgplt1 = plt.imshow(test_imgs[label_index])
		ax1.set_title('Test Image ({})'.format(class_names[i]), size=10)

		for j in range(5):
			ax2 = plt.subplot(2,3,(j+2))
			ax2.axis('off')
			nn_class = class_names[np.where(labels_to_keep == nn_cls[j])[0][0]]
			imgplt2 = plt.imshow(train_imgs[nn_img_indices[j]])
			ax2.set_title('NN: {} , dist: {:0.2f}, class: {}'.format(j+1, nn[j] ,nn_class), size=7)

		plt.show()

def plot_k_error_rates(k_array, dists, train_cls, test_cls):
	e_rates = np.zeros(k_array.shape)
	for k in range(k_array.shape[0]):
		nn, nn_cls, _ = find_nearest_neighbors(k_array[k], dists, train_cls)
		e_rate = compute_error_rate(nn, test_cls, nn_cls)
		e_rates[k] = e_rate
		print('Error rate for k = {} is {}'.format(k_array[k], e_rate))

	plt.plot(k_array, e_rates)
	plt.ylabel('Error Rates')
	plt.xlabel('K')
	plt.show()

def get_dists(filename, train_imgs, test_imgs):
	# size of dists is 4000 x 10000
	if (os.path.isfile(filename)):
		dists = np.load(filename)
		print('Precalculated distances loaded.')
	else:
		print('Calculating distances')
		dists = compute_dists(train_imgs, test_imgs)
		print('Saving distances')
		np.save(filename, dists)
		print('Distances saved')
	return dists


def run_KNN():

	# load CIFAR data
	train_imgs, train_cls, train_names = cifar_loader.load_training_data()
	test_imgs, test_cls, test_names = cifar_loader.load_test_data()
	class_names = np.array(cifar_loader.load_class_names())

	# get relevant class data
	labels_to_keep = np.sort(np.array([1, 2, 3, 4]))
	train_imgs, train_cls = split_sets(train_imgs, train_cls, labels_to_keep)
	test_imgs, test_cls = split_sets(test_imgs, test_cls, labels_to_keep)
	class_names = split_class_names(class_names, labels_to_keep)

	# part a
	# view_examples(train_imgs, train_cls, labels_to_keep, class_names)

	# part b and c
	train_imgs_gray = rgb2gray_luminosity(train_imgs)
	test_imgs_gray = rgb2gray_luminosity(test_imgs)
	dists = get_dists('dists_lum_ssd.npy',train_imgs_gray, test_imgs_gray)



	k_array = np.array([1,2,5,10,20])
	plot_k_error_rates(k_array, dists, train_cls, test_cls)


	# part d
	# plot_nearest_neighbors(labels_to_keep, dists, train_cls, test_cls, test_imgs, train_imgs, class_names)	






if __name__ == '__main__':
    run_KNN()




