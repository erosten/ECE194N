import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator



def get_csv_data(filename):
	return np.loadtxt(filename,dtype='float32',delimiter=',')


def plot_csv(filename, y_label, max_or_min):
	data = get_csv_data(filename)
	epochs = data[:,0]
	acc = data[:,1]
	string = ''
	if (max_or_min == 'max'):
		index = np.argmax(data[:,1])
		string = 'Maximum'
	else:
		index = np.argmin(data[:,1])
		string = 'Minimum'
	# add plot and max/min point
	plt.plot(epochs, acc, linestyle = '--', marker = 'o', color = 'b')
	plt.plot(epochs[index], acc[index], marker = 'o', color='r')
	# add vertical line
	ax = plt.gca()
	y_min, y_max = ax.get_ylim()
	plt.vlines(epochs[index], y_min, acc[index], linestyle='dashed')
	plt.ylim((y_min, y_max))
	# set plot title
	plt.title('{} vs Epoch'.format(y_label))
	# delete x axis ticks that are within one of the max epoch
	x_tick_arr = np.array(list(ax.get_xticks())).astype(int)
	close_indices = np.where(np.logical_or(x_tick_arr==np.int(epochs[index] - 1), x_tick_arr==np.int(epochs[index] + 1)))[0]
	ax.set_xticks(np.delete(x_tick_arr, close_indices))
	# add max index to x axis
	ax.set_xticks(list(ax.get_xticks()) + [epochs[index]])	
	# set plot x/y labels
	plt.xlabel('Epoch')
	plt.ylabel('{}'.format(y_label))
	# add legend
	string = string + ' ' + y_label + ': ' + str(acc[index])
	plt.legend(['_nolegend_', string], loc = 'best')
	# make the max point red
	x_min, x_max = ax.get_xlim()
	plt.xlim((0, epochs[-1:] + 1))
	ax.get_xticklabels()[-1].set_color('red') 

	plt.show()


plot_csv('epoch_loss_train.txt', 'Training Loss', 'min')
plot_csv('epoch_loss_val.txt', 'Validation Loss', 'min')



