import cifar_loader
import matplotlib.pyplot as plt
import numpy as np


def view_examples(train_imgs, train_cls, classes, class_names):
	print(class_names)
	plt.figure(1)
	for i in range(classes.shape[0]):
		label_class_name = class_names[i]
		rand_integer = np.random.randint(0,4999)
		# grab a random example index
		index = np.where(train_cls == classes[i])[0][rand_integer]
		img_example = train_imgs[index]

		ax = plt.subplot(2,classes.shape[0] / 2,i + 1)
		ax.axis('off')
		imgplot = plt.imshow(img_example)
		ax.set_title(label_class_name)
	plt.show()


def main():
	# load cifar data
	train_imgs, train_cls, train_names = cifar_loader.load_training_data()
	class_names = np.array(cifar_loader.load_class_names())
	classes = np.arange(0,10)
	# view examples
	view_examples(train_imgs, train_cls, classes, class_names)




if __name__ == '__main__':
    main()